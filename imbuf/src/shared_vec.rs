use std::{
    marker::PhantomData,
    num::NonZeroU32,
    sync::atomic::{AtomicUsize, Ordering},
    vec::Vec,
};

use crate::{
    ImageChannel, PixelType,
    channel::{ChannelFactory, ImageChannelVTable, UnsafeImageChannel},
    pixel_elements::PixelSize,
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

/// Metadata stored in a Box, with pointer stored in UnsafeImageChannel.data field
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
            let slice_idx = self.slice_idx;
            let _ = shared.slice_refs[slice_idx].fetch_add(1, Ordering::AcqRel) + 1;
            let _ = shared.total_refs.fetch_add(1, Ordering::AcqRel) + 1;
        }
        Self {
            data_ptr: self.data_ptr,
            slice_idx: self.slice_idx,
            start: self.start,
        }
    }
}

// Single generic extern "C" functions with const CHANNELS
// These are instantiated when added to the vtable
unsafe extern "C" fn clone_shared_vec<T: 'static, const CHANNELS: usize>(
    image: &UnsafeImageChannel<T>,
) -> UnsafeImageChannel<T> {
    let metadata = unsafe { &mut *(image.data.cast::<SharedVecMetadata<T, CHANNELS>>()) };

    UnsafeImageChannel {
        ptr: image.ptr,
        width: image.width,
        height: image.height,
        vtable: image.vtable,
        data: Box::into_raw(Box::new(metadata.clone())).cast(),
        pixel_elements: image.pixel_elements,
    }
}

unsafe extern "C" fn make_mut_shared_vec<T: 'static + Clone, const CHANNELS: usize>(
    image: &mut UnsafeImageChannel<T>,
) {
    let metadata = unsafe { &mut *(image.data.cast::<SharedVecMetadata<T, CHANNELS>>()) };
    let data = metadata.data_ptr;
    let slice_idx = metadata.slice_idx;

    let is_unique = unsafe { (*data).slice_refs[slice_idx].load(Ordering::Acquire) == 1 };

    if !is_unique {
        let slice = unsafe {
            std::slice::from_raw_parts(image.ptr, (image.width.get() * image.height.get()) as usize)
        };
        *image = UnsafeImageChannel::new_vec(
            slice.to_vec(),
            image.width,
            image.height,
            image.pixel_elements,
        );
    }
}

pub(crate) extern "C" fn drop_shared_vec<T: 'static, const CHANNELS: usize>(
    image: &mut UnsafeImageChannel<T>,
) {
    unsafe {
        let metadata = Box::from_raw(image.data.cast::<SharedVecMetadata<T, CHANNELS>>());
        let shared = metadata.data_ptr;
        let slice_idx = metadata.slice_idx;
        let _ = (*shared).slice_refs[slice_idx].fetch_sub(1, Ordering::AcqRel) - 1;

        if (*shared).total_refs.fetch_sub(1, Ordering::AcqRel) == 1 {
            drop(Box::from_raw(shared));
        }
    };
}

struct SharedVecFactory<T: 'static, const CHANNELS: usize>(PhantomData<(T, [(); CHANNELS])>);

// Implement ChannelFactory with const VTABLE using associated const
// PhantomData makes this type unique for each T and CHANNELS combination
impl<T: 'static + Clone, const CHANNELS: usize> ChannelFactory<T>
    for SharedVecFactory<T, CHANNELS>
{
    const VTABLE: &'static ImageChannelVTable<T> = {
        &ImageChannelVTable {
            clone: clone_shared_vec::<T, CHANNELS>,
            make_mut: make_mut_shared_vec::<T, CHANNELS>,
            drop: drop_shared_vec::<T, CHANNELS>,
        }
    };
}

/// Create `ImageChannels` from a Vec, sharing the underlying storage
/// Note: T: Clone is required for vtable creation, but clone/drop don't actually need it
pub fn create_shared_channels<TP: PixelType, const CHANNELS: usize>(
    vec: Vec<TP::Primitive>,
    sizes: [(NonZeroU32, NonZeroU32); CHANNELS],
) -> [ImageChannel<TP>; CHANNELS]
where
    TP::Primitive: Clone,
{
    assert_eq!(
        vec.len(),
        sizes.iter().fold(0, |acc, i| acc
            + i.0.get() as usize
                * i.1.get() as usize
                * TP::PixelSize::default().get().get() as usize)
    );
    // Create SharedVecData
    let mut base = vec.as_ptr();
    let shared_data = Box::new(SharedVecData::<TP::Primitive, CHANNELS>::new(vec));
    let data_ptr = Box::into_raw(shared_data);

    // Create ImageChannels for each slice
    std::array::from_fn(|i| {
        let (width, height) = sizes[i];
        let start = width.get() as usize
            * height.get() as usize
            * TP::PixelSize::default().get().get() as usize;

        let metadata = Box::new(SharedVecMetadata::<TP::Primitive, CHANNELS> {
            data_ptr,
            slice_idx: i,
            start,
        });
        let metadata_ptr = Box::into_raw(metadata);
        let vtable =
            <SharedVecFactory<TP::Primitive, CHANNELS> as ChannelFactory<TP::Primitive>>::VTABLE;

        let ptr = base;

        unsafe {
            base = base.add(start);
            ImageChannel::from_unsafe_internal(UnsafeImageChannel::new_with_vtable(
                ptr,
                width,
                height,
                TP::PixelSize::default().get(),
                vtable,
                metadata_ptr.cast(),
            ))
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
}
