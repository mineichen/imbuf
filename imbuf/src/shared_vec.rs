use std::{
    marker::PhantomData,
    num::{NonZeroU8, NonZeroU32},
    sync::atomic::{AtomicUsize, Ordering},
    vec::Vec,
};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, ImageChannel, InvalidLenError, PixelType,
    PixelTypePrimitive,
    channel::{ChannelFactory, ImageChannelVTable, bytes_per_primitive},
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
            bytes_per_primitive: bytes_per_primitive::<T>,
        }
    };
}

/// Creates channels, which share the planar `vec` (all pixels of the first channel, then the
/// second...). Each channel can have its own size (e.g. the smaller U and V planes of YUV).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "For channels of different sizes (e.g. YUV), images use the unchecked version"
    )
)]
pub(crate) fn create_shared_channels<TP: PixelType, const CHANNELS: usize>(
    vec: Vec<TP::Primitive>,
    sizes: [(NonZeroU32, NonZeroU32); CHANNELS],
) -> Result<[ImageChannel<TP>; CHANNELS], InvalidLenError> {
    ensure_valid_shared_len(&sizes, TP::ELEMENTS, vec.len())?;
    // Safety: The length is checked above
    Ok(unsafe { create_shared_channels_unchecked(vec, sizes) })
}

/// Unchecked version of [`create_shared_channels`]
///
/// # Safety
/// The lengths of the channels (`width * height * TP::ELEMENTS`) and their sum mustn't overflow
/// `usize` and the sum has to match `vec.len()` (see [`ensure_valid_shared_len`])
pub(crate) unsafe fn create_shared_channels_unchecked<TP: PixelType, const CHANNELS: usize>(
    mut vec: Vec<TP::Primitive>,
    sizes: [(NonZeroU32, NonZeroU32); CHANNELS],
) -> [ImageChannel<TP>; CHANNELS] {
    let _non_zero = const { crate::unwrap_usize_to_nonzero_u8(CHANNELS) };
    debug_assert_eq!(
        Ok(()),
        ensure_valid_shared_len(&sizes, TP::ELEMENTS, vec.len())
    );
    let base = vec.as_mut_ptr();
    let data_ptr = Box::into_raw(Box::new(SharedVecData::<TP::Primitive, CHANNELS>::new(vec)));
    let vtable =
        <SharedVecFactory<TP::Primitive, CHANNELS> as ChannelFactory<TP::Primitive>>::VTABLE;
    let mut start = 0;

    std::array::from_fn(|i| {
        let (width, height) = sizes[i];
        let metadata = Box::new(SharedVecMetadata::<TP::Primitive, CHANNELS> {
            data_ptr,
            slice_idx: i,
            start,
        });
        // Safety: The slice is within the Vec (precondition). The Vec keeps the slice alive until
        // the vtable drops it and the vtable is built for TP::Primitive with TP::ELEMENTS
        let channel = unsafe {
            DynamicImageChannel::from_raw_parts(
                base.add(start).cast(),
                width,
                height,
                TP::ELEMENTS,
                vtable,
                Box::into_raw(metadata).cast(),
            )
        };
        // The lengths sum up to the length of the Vec (precondition), so this cannot overflow
        start += channel.len_flat();
        // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
        unsafe { ImageChannel::from_dynamic_unchecked(channel) }
    })
}

/// Checks that the lengths of the channels (`width * height * pixel_elements`) and their sum
/// don't overflow `usize` and that the sum matches `len`
fn ensure_valid_shared_len(
    sizes: &[(NonZeroU32, NonZeroU32)],
    pixel_elements: NonZeroU8,
    len: usize,
) -> Result<(), InvalidLenError> {
    let mut total: usize = 0;
    for &(width, height) in sizes {
        let channel_len = DynamicImageChannelView::flat_len_checked(
            width,
            height,
            pixel_elements,
            NonZeroU8::MIN,
        )?;
        total = total
            .checked_add(channel_len)
            .ok_or(InvalidLenError::BufferMismatch {
                expected: len,
                actual: usize::MAX,
            })?;
    }
    if total == len {
        Ok(())
    } else {
        Err(InvalidLenError::BufferMismatch {
            expected: total,
            actual: len,
        })
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use testresult::TestResult;

    use super::*;

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

    #[test]
    fn test_shared_vec_make_mut() -> TestResult {
        let vec = vec![0u8, 1, 2, 3, 4, 5];
        let orig_ptr = vec.as_ptr();
        let mut channels = create_shared_channels::<u8, 3>(vec, [(NZ_2, NZ_1); 3])?;
        let mutbuf = channels[0].make_mut();
        assert_eq!(mutbuf.as_ptr(), orig_ptr);
        Ok(())
    }

    #[test]
    fn non_unique_clone_make_mut() -> TestResult {
        let vec = vec![0u8, 1, 2, 3, 4, 5];
        let orig_ptr = vec.as_ptr();
        let mut channels = create_shared_channels::<u8, 3>(vec, [(NZ_2, NZ_1); 3])?;
        let clone = channels[0].clone();
        let mutbuf = channels[0].make_mut();
        assert_eq!(clone.buffer().as_ptr(), orig_ptr);
        assert_ne!(mutbuf.as_ptr(), orig_ptr);
        Ok(())
    }

    #[test]
    fn unique_after_dropped_clone_make_mut() -> TestResult {
        let vec = vec![0u8, 1, 2, 3, 4, 5];
        let orig_ptr = vec.as_ptr();
        let mut channels = create_shared_channels::<u8, 3>(vec, [(NZ_2, NZ_1); 3])?;
        drop(channels[0].clone());
        let mutbuf = channels[0].make_mut();
        assert_eq!(mutbuf.as_ptr(), orig_ptr);
        Ok(())
    }

    #[test]
    fn test_shared_vec_basic() -> TestResult {
        let vec = vec![0u8, 1, 2, 3, 4, 5];
        // Get the address of the original Vec's data before it's moved
        let original_ptr = vec.as_ptr();

        let len_per_channel = 2;

        let channels = create_shared_channels::<u8, 3>(vec, [(NZ_2, NZ_1); 3])?;

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
        Ok(())
    }

    #[test]
    fn test_shared_vec_clone() -> TestResult {
        let vec = vec![0u8, 1, 2, 3];

        let channels = create_shared_channels::<u8, 2>(vec, [(NZ_2, NZ_1); 2])?;
        let channel1_clone = channels[0].clone();

        assert_eq!(
            channels[0].buffer().as_ptr(),
            channel1_clone.buffer().as_ptr()
        );
        Ok(())
    }

    #[test]
    fn non_unique_make_mut_copies_all_pixel_elements() -> TestResult {
        let vec = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let mut channels = create_shared_channels::<[u8; 2], 2>(vec, [(NZ_2, NZ_1); 2])?;
        let clone = channels[1].clone();

        channels[1].make_mut()[0] = [8, 9];
        assert_eq!(channels[1].buffer(), &[[8, 9], [6, 7]]);
        assert_eq!(clone.buffer(), &[[4, 5], [6, 7]]);
        assert_eq!(channels[0].buffer(), &[[0, 1], [2, 3]]);
        Ok(())
    }

    #[test]
    fn channels_of_different_sizes() -> TestResult {
        // E.g. YUV 4:2:0, where U and V have a quarter of the pixels of Y
        let vec = vec![0u16, 1, 2, 3, 4, 5];
        let channels =
            create_shared_channels::<u16, 3>(vec, [(NZ_2, NZ_2), (NZ_1, NZ_1), (NZ_1, NZ_1)])?;
        assert_eq!(channels[0].buffer(), &[0, 1, 2, 3]);
        assert_eq!(channels[1].buffer(), &[4]);
        assert_eq!(channels[2].buffer(), &[5]);
        Ok(())
    }

    #[test]
    fn channel_which_does_not_fit_fails() {
        // The second channel requires 4 elements, but only 2 are left
        assert_matches!(
            create_shared_channels::<[u8; 2], 2>(vec![0; 6], [(NZ_2, NZ_1); 2]),
            Err(InvalidLenError::BufferMismatch {
                expected: 8,
                actual: 6
            })
        );
    }

    #[test]
    fn remaining_elements_fail() {
        assert_matches!(
            create_shared_channels::<[u8; 2], 2>(vec![0; 10], [(NZ_2, NZ_1); 2]),
            Err(InvalidLenError::BufferMismatch {
                expected: 8,
                actual: 10
            })
        );
    }

    #[test]
    fn overflowing_channel_fails() {
        assert_matches!(
            create_shared_channels::<[u8; 255], 1>(vec![], [(NonZeroU32::MAX, NonZeroU32::MAX)]),
            Err(InvalidLenError::Overflow(_))
        );
    }
}
