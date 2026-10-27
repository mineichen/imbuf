//! Storages for borrowed buffers

use std::num::{NonZeroU8, NonZeroU32};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, PixelTypePrimitive,
    channel::{ChannelFactory, ImageChannelVTable, UnsafeImageChannel, primitive_size},
};

/// Shared borrow of a buffer. `make_mut` copies the buffer into a Vec.
struct SliceFactory;

/// Exclusive borrow of a buffer. `clone` copies the buffer.
///
/// `data` is either null (the buffer is exclusive) or points to the borrowed
/// `DynamicImageChannelView`, whose `make_mut` is called before writing into its buffer.
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
    let raw = UnsafeImageChannel::new_with_vtable(
        ptr.cast(),
        width,
        height,
        pixel_elements,
        <SliceFactory as ChannelFactory<T>>::VTABLE,
        std::ptr::null_mut(),
    );
    // Safety: Forwarded to the caller
    unsafe { DynamicImageChannelView::from_unsafe(raw) }
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
    let raw = UnsafeImageChannel::new_with_vtable(
        ptr.cast_const().cast(),
        width,
        height,
        pixel_elements,
        <SliceMutFactory as ChannelFactory<T>>::VTABLE,
        original.cast(),
    );
    // Safety: Forwarded to the caller
    unsafe { DynamicImageChannelView::from_unsafe(raw) }
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for SliceFactory {
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable {
        clone: clone_slice_channel,
        make_mut: copy_into_vec::<T>,
        drop: noop_drop,
        primitive_size: primitive_size::<T>,
    };
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for SliceMutFactory {
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable {
        clone: crate::arc::clone_slice_into_arc_channel::<T>,
        make_mut: make_original_mut,
        drop: noop_drop,
        primitive_size: primitive_size::<T>,
    };
}

/// Shares the borrowed buffer, which is valid for `'a`
unsafe extern "C-unwind" fn clone_slice_channel<'a>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    let raw = DynamicImageChannelView::get_unsafe(image);
    let raw = UnsafeImageChannel::new_with_vtable(
        raw.ptr,
        raw.width,
        raw.height,
        raw.pixel_elements,
        raw.vtable,
        raw.data,
    );
    // Safety: The buffer is valid and immutable for 'a
    unsafe { DynamicImageChannelView::from_unsafe(raw) }
}

unsafe extern "C-unwind" fn copy_into_vec<T: PixelTypePrimitive>(
    image: &mut DynamicImageChannelView<'_>,
) {
    // Safety: The vtable is built for T
    let copy = unsafe { image.buffer_flat_unchecked::<T>() }.to_vec();
    *image =
        DynamicImageChannel::from_vec(copy, image.width(), image.height(), image.pixel_elements());
}

unsafe extern "C-unwind" fn make_original_mut(image: &mut DynamicImageChannelView<'_>) {
    // Safety: The layout of the buffer doesn't change
    let raw = unsafe { DynamicImageChannelView::get_unsafe_mut(image) };
    let original = raw.data.cast::<DynamicImageChannelView<'_>>();
    if !original.is_null() {
        // Safety: The original channel is exclusively borrowed by this channel
        let original = unsafe { &mut *original };
        original.make_exclusive();
        raw.ptr = DynamicImageChannelView::get_unsafe(original).ptr;
    }
}

unsafe extern "C-unwind" fn noop_drop(_image: &mut DynamicImageChannelView<'_>) {
    // The buffer is borrowed
}
