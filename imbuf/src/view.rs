//! Storages for borrowed buffers

use std::num::{NonZeroU8, NonZeroU32};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, PixelTypePrimitive,
    vtable::{ChannelFactory, ImageChannelVTable},
};

/// Shared borrow of a buffer. `make_mut` copies the buffer into an `Arc`.
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
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable::new::<T>(
        clone_slice_channel,
        copy_into_arc::<T>,
        borrowed_immutably,
        noop_drop,
    );
}

/// `try_make_mut` of [`SliceFactory`]: The buffer is never written, as it's borrowed immutably
unsafe extern "C-unwind" fn borrowed_immutably(_: &mut DynamicImageChannelView<'_>) -> bool {
    false
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for SliceMutFactory {
    const VTABLE: &'static ImageChannelVTable = {
        unsafe extern "C-unwind" fn make_original_mut<T: PixelTypePrimitive>(
            image: &mut DynamicImageChannelView<'_>,
        ) {
            // Safety: The channel was created by `slice_mut_channel`
            if let Some(original) = unsafe { original(image) } {
                original.make_exclusive();
                // Safety: `original` is the original of `image`
                unsafe { reborrow_original::<T>(image, original) };
            }
        }

        unsafe extern "C-unwind" fn try_make_original_mut<T: PixelTypePrimitive>(
            image: &mut DynamicImageChannelView<'_>,
        ) -> bool {
            // Safety: The channel was created by `slice_mut_channel`
            let Some(original) = (unsafe { original(image) }) else {
                return true;
            };
            let exclusive = original.try_make_exclusive();
            // Safety: `original` is the original of `image`
            unsafe { reborrow_original::<T>(image, original) };
            exclusive
        }

        &ImageChannelVTable::new::<T>(
            crate::arc::clone_slice_into_arc_channel::<T>,
            make_original_mut::<T>,
            try_make_original_mut::<T>,
            noop_drop,
        )
    };
}

/// The original channel, which a channel of [`SliceMutFactory`] borrows exclusively. `None`, if
/// it borrows a mutable slice, whose buffer is exclusively borrowed for the lifetime of the
/// channel, which is only written through, never reallocated or freed.
///
/// # Safety
/// `image` has to be created by [`slice_mut_channel`]. The original mustn't be used through
/// `image` while the result is alive.
unsafe fn original<'b>(
    image: &DynamicImageChannelView<'_>,
) -> Option<&'b mut DynamicImageChannelView<'b>> {
    let raw = DynamicImageChannelView::raw_data(image);
    // Safety: The original channel is exclusively borrowed by this channel
    (!raw.is_null()).then(|| unsafe { &mut *raw.cast::<DynamicImageChannelView<'_>>() })
}

/// Points `image` to the buffer of `original` again. Its `make_mut` and `try_make_mut` could have
/// replaced the buffer. Replacing `image` preserves the borrowed original, as the vtable drops
/// nothing and `data` points to it again.
///
/// # Safety
/// `image` has to be created by [`slice_mut_channel`] for `original` (see [`original`])
unsafe fn reborrow_original<T: PixelTypePrimitive>(
    image: &mut DynamicImageChannelView<'_>,
    original: &mut DynamicImageChannelView<'_>,
) {
    // Safety: The original is exclusively borrowed by `image`, which is replaced
    *image = unsafe {
        slice_mut_channel::<T>(
            DynamicImageChannelView::raw_ptr(original).cast_mut().cast(),
            original.width(),
            original.height(),
            original.pixel_elements(),
            original,
        )
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
        DynamicImageChannel::new_arc_flat_unchecked(
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
