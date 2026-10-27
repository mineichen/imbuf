use std::{
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32},
    ptr,
    sync::Arc,
};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, PixelTypePrimitive,
    channel::{
        ChannelFactory, ImageChannelVTable, UnsafeImageChannel, calc_pixel_len_flat, primitive_size,
    },
};

struct ArcFactory;

impl DynamicImageChannel {
    /// In contrast to the public constructors, `T` doesn't have to be `Pod`. Don't expose the
    /// result as `DynamicImageChannel`, unless `T` is `Pod`
    ///
    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height and `pixel_elements`.
    pub(crate) fn from_arc<T: PixelTypePrimitive>(
        input: Arc<[T]>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Self {
        let len = input.len();
        assert_eq!(
            len,
            calc_pixel_len_flat(width, height, pixel_elements),
            "Incompatible Buffer-Size"
        );

        let ptr = Arc::into_raw(input).cast::<u8>();
        let unsafe_ch = UnsafeImageChannel::new_with_vtable(
            ptr,
            width,
            height,
            pixel_elements,
            <ArcFactory as ChannelFactory<T>>::VTABLE,
            ptr::without_provenance_mut(len),
        );
        unsafe { DynamicImageChannel::from_unsafe(unsafe_ch) }
    }
}

/// Borrows the `Arc` of a channel created by `from_arc` without taking ownership
unsafe fn arc_view<T>(image: &DynamicImageChannelView<'_>) -> ManuallyDrop<Arc<[T]>> {
    let raw = DynamicImageChannelView::get_unsafe(image);
    let ptr = ptr::slice_from_raw_parts(raw.ptr.cast::<T>(), raw.data as usize);
    ManuallyDrop::new(unsafe { Arc::from_raw(ptr) })
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for ArcFactory {
    const VTABLE: &'static ImageChannelVTable = {
        unsafe extern "C-unwind" fn make_mut<T: Clone>(image: &mut DynamicImageChannelView<'_>) {
            let mut arc = unsafe { arc_view::<T>(image) };

            if Arc::get_mut(&mut arc).is_none() {
                let new_data = Arc::<[T]>::from(&arc[..]);
                ManuallyDrop::into_inner(arc);

                unsafe { DynamicImageChannelView::get_unsafe_mut(image) }.ptr =
                    Arc::into_raw(new_data).cast::<u8>();
            }
        }
        unsafe extern "C-unwind" fn clear_arc_channel<T>(image: &mut DynamicImageChannelView<'_>) {
            drop(ManuallyDrop::into_inner(unsafe { arc_view::<T>(image) }));
        }

        unsafe extern "C-unwind" fn clone_arc_channel<'a, T: PixelTypePrimitive>(
            image: &DynamicImageChannelView<'a>,
        ) -> DynamicImageChannelView<'a> {
            let arc = unsafe { arc_view::<T>(image) };
            DynamicImageChannel::from_arc(
                Arc::clone(&arc),
                image.width(),
                image.height(),
                image.pixel_elements(),
            )
        }

        &ImageChannelVTable {
            drop: clear_arc_channel::<T>,
            clone: clone_arc_channel::<T>,
            make_mut: make_mut::<T>,
            primitive_size: primitive_size::<T>,
        }
    };
}

pub(crate) unsafe extern "C-unwind" fn clone_slice_into_arc_channel<'a, T: PixelTypePrimitive>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    DynamicImageChannel::from_arc(
        // Safety: The vtable is built for T
        Arc::from(unsafe { image.buffer_flat_unchecked::<T>() }),
        image.width(),
        image.height(),
        image.pixel_elements(),
    )
}
