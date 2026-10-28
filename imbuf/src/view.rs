//! Storages for borrowed buffers

use std::num::{NonZeroU8, NonZeroU32};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, PixelTypePrimitive,
    channel::{ChannelFactory, ImageChannelVTable, primitive_size},
};

/// Shared borrow of a buffer. `make_mut` copies the buffer into a Vec.
struct SliceFactory;

/// Exclusive borrow of a buffer. `clone` copies the buffer.
///
/// `data` is either null (the buffer is exclusive) or points to the borrowed
/// `GenericDynamicImageChannel`, whose `make_mut` is called before writing into its buffer.
struct SliceMutFactory;

/// # Safety
/// `ptr` has to point to `width * height * pixel_elements` initialized values of `T`, which are
/// not mutated during `'a`.
pub(crate) unsafe fn slice_channel<'a, T: PixelTypePrimitive>(
    ptr: *const T,
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
) -> DynamicImageChannelView<'a> {
    // Safety: Forwarded to the caller. The drop is a no-op, as the buffer is borrowed
    unsafe {
        DynamicImageChannelView::from_raw_parts(
            ptr.cast(),
            width,
            height,
            pixel_elements,
            <SliceFactory as ChannelFactory<T>>::VTABLE,
            std::ptr::null_mut(),
        )
    }
}

/// # Safety
/// `ptr` has to point to `width * height * pixel_elements` initialized values of `T`, which are
/// exclusively borrowed for `'a`. If `original` is not null, it has to be exclusively borrowed
/// for `'a` and `ptr` has to be its buffer, which is only written after calling its `make_mut`.
pub(crate) unsafe fn slice_mut_channel<'a, T: PixelTypePrimitive>(
    ptr: *mut T,
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
    original: *mut DynamicImageChannelView<'_>,
) -> DynamicImageChannelView<'a> {
    // Safety: Forwarded to the caller. The drop is a no-op, as the buffer is borrowed
    unsafe {
        DynamicImageChannelView::from_raw_parts(
            ptr.cast_const().cast(),
            width,
            height,
            pixel_elements,
            <SliceMutFactory as ChannelFactory<T>>::VTABLE,
            original.cast(),
        )
    }
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for SliceFactory {
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable {
        clone: clone_slice_channel,
        make_mut: copy_into_arc::<T>,
        drop: noop_drop,
        primitive_size: primitive_size::<T>,
    };
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for SliceMutFactory {
    const VTABLE: &'static ImageChannelVTable = {
        unsafe extern "C-unwind" fn make_original_mut<T: PixelTypePrimitive>(
            image: &mut DynamicImageChannelView<'_>,
        ) {
            let raw = DynamicImageChannelView::raw_data(&*image);
            let created_from_mut_slice = raw.is_null();
            if created_from_mut_slice {
                // The buffer is exclusively borrowed for the lifetime of the channel, which is
                // only written through, never reallocated or freed
                return;
            }
            let original = raw.cast::<DynamicImageChannelView<'_>>();
            // Safety: The original channel is exclusively borrowed by this channel
            let original = unsafe { &mut *original };
            original.make_exclusive();
            // The vtable of the original could have replaced the buffer, so the pointer is read
            // after `make_exclusive`. Replacing `image` preserves the borrowed original, as the
            // vtable drops nothing and `data` points to it again.
            *image = unsafe {
                crate::view::slice_mut_channel::<T>(
                    DynamicImageChannelView::raw_ptr(original).cast_mut().cast(),
                    original.width(),
                    original.height(),
                    original.pixel_elements(),
                    original,
                )
            };
        }

        &ImageChannelVTable {
            clone: crate::arc::clone_slice_into_arc_channel::<T>,
            make_mut: make_original_mut::<T>,
            drop: noop_drop,
            primitive_size: primitive_size::<T>,
        }
    };
}

/// Shares the borrowed buffer, which is valid for `'a`
unsafe extern "C-unwind" fn clone_slice_channel<'a>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    // Safety: The buffer is valid and immutable for 'a
    unsafe {
        DynamicImageChannelView::from_raw_parts(
            DynamicImageChannelView::raw_ptr(image),
            image.width(),
            image.height(),
            image.pixel_elements(),
            DynamicImageChannelView::raw_vtable(image),
            DynamicImageChannelView::raw_data(image),
        )
    }
}

unsafe extern "C-unwind" fn copy_into_arc<T: PixelTypePrimitive>(
    image: &mut DynamicImageChannelView<'_>,
) {
    // Safety: The vtable is built for T
    let copy = unsafe { image.buffer_flat_unchecked::<T>() };
    *image = unsafe {
        DynamicImageChannel::from_arc_flat_unchecked(
            copy.into(),
            image.width(),
            image.height(),
            image.pixel_elements(),
        )
    };
}

unsafe extern "C-unwind" fn noop_drop(_image: &mut DynamicImageChannelView<'_>) {
    // The buffer is borrowed
}
