use std::{
    marker::PhantomData,
    num::NonZeroU32,
    sync::atomic::{AtomicUsize, Ordering},
    vec::Vec,
};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, ImageChannel, PixelType, PixelTypePrimitive,
    channel::{ChannelFactory, ImageChannelVTable, primitive_size},
    dynamic::flat_len_checked,
};

/// Internal structure that holds a Vec (as raw parts) and reference counts
/// This allows multiple `ImageChannels` to share the same Vec
#[repr(C)]
pub struct SharedVecData<T, const CHANNELS: usize> {
    /// Pointer to the start of the Vec data
    vec: Vec<T>,
    /// Total number of `ImageChannels` using this `SharedVec` (global atomic)
    total_refs: AtomicUsize,
    /// Per-slice reference counts (one per slice/channel) - used to detect if mutual borrowing is ok
    slice_refs: [AtomicUsize; CHANNELS],
}

impl<T, const CHANNELS: usize> SharedVecData<T, CHANNELS> {
    fn new(vec: Vec<T>) -> Self {
        Self {
            vec,
            total_refs: AtomicUsize::new(CHANNELS),
            slice_refs: std::array::from_fn(|_| AtomicUsize::new(1)),
        }
    }
}

/// Metadata stored in a Box, with pointer stored in the `data` field of the channel
/// Minimal: only stores what's needed to access the shared data
#[repr(C)]
struct SharedVecMetadata<T, const CHANNELS: usize> {
    /// Pointer to the `SharedVecData`
    data_ptr: *mut SharedVecData<T, CHANNELS>,
    /// Index of this slice (to access the correct `slice_refs` in `SharedVecData`)
    slice_idx: usize,
    /// Start offset in the Vec for this slice
    start: usize,
}

impl<T, const CHANNELS: usize> Clone for SharedVecMetadata<T, CHANNELS> {
    fn clone(&self) -> Self {
        unsafe {
            let shared = &(*self.data_ptr);
            shared.slice_refs[self.slice_idx].fetch_add(1, Ordering::AcqRel);
            shared.total_refs.fetch_add(1, Ordering::AcqRel);
        }
        Self {
            data_ptr: self.data_ptr,
            slice_idx: self.slice_idx,
            start: self.start,
        }
    }
}

/// Borrows the metadata of a channel created by `create_shared_channels`
unsafe fn metadata<'b, T, const CHANNELS: usize>(
    image: &'b DynamicImageChannelView<'_>,
) -> &'b SharedVecMetadata<T, CHANNELS> {
    let data = DynamicImageChannelView::raw_data(image);
    unsafe { &*(data.cast::<SharedVecMetadata<T, CHANNELS>>()) }
}

// Single generic extern "C-unwind" functions with const CHANNELS
// These are instantiated when added to the vtable
unsafe extern "C-unwind" fn clone_shared_vec<'a, T: 'static, const CHANNELS: usize>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    let metadata = unsafe { metadata::<T, CHANNELS>(image) };
    // Safety: The shared buffer stays valid until the vtable drops it and the vtable is built
    // for `T` with `CHANNELS`
    unsafe {
        DynamicImageChannel::from_raw_parts(
            DynamicImageChannelView::raw_ptr(image),
            image.width(),
            image.height(),
            image.pixel_elements(),
            DynamicImageChannelView::raw_vtable(image),
            Box::into_raw(Box::new(metadata.clone())).cast(),
        )
    }
}

unsafe extern "C-unwind" fn make_mut_shared_vec<T: PixelTypePrimitive, const CHANNELS: usize>(
    channel: &mut DynamicImageChannelView<'_>,
) {
    let metadata = unsafe { metadata::<T, CHANNELS>(channel) };
    let is_unique =
        unsafe { (*metadata.data_ptr).slice_refs[metadata.slice_idx].load(Ordering::Acquire) == 1 };

    if !is_unique {
        // Safety: The vtable is built for T
        let copy = unsafe { channel.buffer_flat_unchecked::<T>() }.to_vec();
        // Assigning drops the previous channel, which releases its share of the Vec
        // Safety: Since we got a DynamicImageChannelView as input, sizes are valid
        *channel = unsafe {
            DynamicImageChannel::from_vec_flat_unchecked(
                copy,
                channel.width(),
                channel.height(),
                channel.pixel_elements(),
            )
        };
    }
}

pub(crate) unsafe extern "C-unwind" fn drop_shared_vec<T: 'static, const CHANNELS: usize>(
    image: &mut DynamicImageChannelView<'_>,
) {
    let data = DynamicImageChannelView::raw_data(image);
    unsafe {
        let metadata = Box::from_raw(data.cast::<SharedVecMetadata<T, CHANNELS>>());
        let shared = metadata.data_ptr;
        (*shared).slice_refs[metadata.slice_idx].fetch_sub(1, Ordering::AcqRel);

        if (*shared).total_refs.fetch_sub(1, Ordering::AcqRel) == 1 {
            drop(Box::from_raw(shared));
        }
    };
}

struct SharedVecFactory<T: 'static, const CHANNELS: usize>(PhantomData<(T, [(); CHANNELS])>);

// Implement ChannelFactory with const VTABLE using associated const
// PhantomData makes this type unique for each T and CHANNELS combination
impl<T: PixelTypePrimitive, const CHANNELS: usize> ChannelFactory<T>
    for SharedVecFactory<T, CHANNELS>
{
    const VTABLE: &'static ImageChannelVTable = {
        &ImageChannelVTable {
            clone: clone_shared_vec::<T, CHANNELS>,
            make_mut: make_mut_shared_vec::<T, CHANNELS>,
            drop: drop_shared_vec::<T, CHANNELS>,
            primitive_size: primitive_size::<T>,
        }
    };
}

/// Create `ImageChannels` from a Vec, sharing the underlying storage
/// #Panics
/// If any size exceeds usize::MAX
pub fn create_shared_channels<TP: PixelType, const CHANNELS: usize>(
    mut vec: Vec<TP::Primitive>,
    sizes: [(NonZeroU32, NonZeroU32); CHANNELS],
) -> [ImageChannel<TP>; CHANNELS] {
    // Panic, if the multiplication of the dimensions or the sum overflows, instead of comparing
    // against wrapped values

    let lens: [usize; CHANNELS] = sizes.map(|(width, height)| {
        flat_len_checked(width, height, TP::ELEMENTS).expect("The buffer length overflows usize")
    });
    let total = lens
        .iter()
        // Panic, if the sum overflows, instead of comparing against a wrapped value
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .expect("The buffer length overflows usize");
    assert_eq!(total, vec.len(), "Incompatible Buffer-Size");
    // Create SharedVecData
    let base = vec.as_mut_ptr();
    let shared_data = Box::new(SharedVecData::<TP::Primitive, CHANNELS>::new(vec));
    let data_ptr = Box::into_raw(shared_data);
    let mut start = 0;

    // Create ImageChannels for each slice
    std::array::from_fn(|i| {
        let (width, height) = sizes[i];
        let metadata = Box::new(SharedVecMetadata::<TP::Primitive, CHANNELS> {
            data_ptr,
            slice_idx: i,
            start,
        });
        let metadata_ptr = Box::into_raw(metadata);
        let vtable =
            <SharedVecFactory<TP::Primitive, CHANNELS> as ChannelFactory<TP::Primitive>>::VTABLE;

        // Safety: The slice is within the Vec, as the total length is checked above
        let ptr = unsafe { base.add(start) };
        // All lengths are part of the total, which is checked above to be the Vec length, so
        // adding them and the subsequent additions of `start` cannot overflow
        let len = lens[i];
        start += len;

        // Safety: The Vec keeps the slice alive until the vtable drops it and the vtable is
        // built for TP::Primitive with TP::ELEMENTS
        unsafe {
            let channel = DynamicImageChannel::from_raw_parts(
                ptr.cast(),
                width,
                height,
                TP::ELEMENTS,
                vtable,
                metadata_ptr.cast(),
            );
            ImageChannel::from_dynamic_unchecked(channel)
        }
    })
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_shared_vec_make_mut() {
        let vec = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
        let orig_ptr = vec.as_ptr();
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut channels = create_shared_channels::<u8, 3>(vec, [(width, height); 3]);
        let mutbuf = channels[0].make_mut();
        assert_eq!(mutbuf.as_ptr(), orig_ptr);
    }

    #[test]
    fn non_unique_clone_make_mut() {
        let vec = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
        let orig_ptr = vec.as_ptr();
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut channels = create_shared_channels::<u8, 3>(vec, [(width, height); 3]);
        let clone = channels[0].clone();
        let mutbuf = channels[0].make_mut();
        assert_eq!(clone.buffer().as_ptr(), orig_ptr);
        assert_ne!(mutbuf.as_ptr(), orig_ptr);
    }

    #[test]
    fn unique_after_dropped_clone_make_mut() {
        let vec = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
        let orig_ptr = vec.as_ptr();
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut channels = create_shared_channels::<u8, 3>(vec, [(width, height); 3]);
        drop(channels[0].clone());
        let mutbuf = channels[0].make_mut();
        assert_eq!(mutbuf.as_ptr(), orig_ptr);
    }

    #[test]
    fn test_shared_vec_basic() {
        let vec = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
        // Get the address of the original Vec's data before it's moved
        let original_ptr = vec.as_ptr();

        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let len_per_channel = 2;

        let channels = create_shared_channels::<u8, 3>(vec, [(width, height); 3]);

        // Verify that channels point to the correct offsets in the original Vec
        assert_eq!(
            channels[0].buffer().as_ptr(),
            original_ptr,
            "Channel 0 should point to the start of the original Vec"
        );
        assert_eq!(
            channels[1].buffer().as_ptr(),
            unsafe { original_ptr.add(len_per_channel) },
            "Channel 1 should point to offset 2 in the original Vec"
        );
        assert_eq!(
            channels[2].buffer().as_ptr(),
            unsafe { original_ptr.add(len_per_channel * 2) },
            "Channel 2 should point to offset 4 in the original Vec"
        );
    }

    #[test]
    fn test_shared_vec_clone() {
        let vec = vec![0u8, 1u8, 2u8, 3u8];
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();

        let channels = create_shared_channels::<u8, 2>(vec, [(width, height); 2]);
        let channel1_clone = channels[0].clone();

        assert_eq!(
            channels[0].buffer().as_ptr(),
            channel1_clone.buffer().as_ptr()
        );
    }

    #[test]
    fn non_unique_make_mut_copies_all_pixel_elements() {
        let vec = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut channels = create_shared_channels::<[u8; 2], 2>(vec, [(width, height); 2]);
        let clone = channels[1].clone();

        channels[1].make_mut()[0] = [8, 9];
        assert_eq!(channels[1].buffer(), &[[8, 9], [6, 7]]);
        assert_eq!(clone.buffer(), &[[4, 5], [6, 7]]);
        assert_eq!(channels[0].buffer(), &[[0, 1], [2, 3]]);
    }
}
