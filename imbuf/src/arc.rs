use std::{
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32},
    sync::Arc,
};

use crate::channel::{ChannelFactory, ImageChannelVTable, UnsafeImageChannel, calc_pixel_len_flat};

struct ArcFactory;

impl<T> UnsafeImageChannel<T>
where
    T: Clone + 'static,
{
    #[must_use]
    /// # Panics
    /// Panics if the buffer size is not compatible with the `width`, `height` and `pixel_elements`.
    pub fn new_arc(
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

        let ptr = Arc::into_raw(input).cast::<T>();
        let vtable = <ArcFactory as ChannelFactory<T>>::VTABLE;
        unsafe {
            Self::new_with_vtable(
                ptr,
                width,
                height,
                pixel_elements,
                vtable,
                std::ptr::without_provenance_mut(len),
            )
        }
    }
}

impl<T: 'static + Clone> ChannelFactory<T> for ArcFactory {
    const VTABLE: &'static ImageChannelVTable<T> = {
        unsafe extern "C" fn make_mut<T: Clone>(image: &mut UnsafeImageChannel<T>) {
            let mut arc = ManuallyDrop::new(unsafe {
                let ptr = std::ptr::slice_from_raw_parts(image.ptr, image.data as usize);
                Arc::<[T]>::from_raw(ptr)
            });

            if Arc::get_mut(&mut arc).is_none() {
                let new_data = Arc::<[T]>::from(&arc[..]);
                ManuallyDrop::into_inner(arc);

                image.ptr = Arc::into_raw(new_data).cast::<T>();
            }
        }
        extern "C" fn clear_arc_channel<T: Clone>(image: &mut UnsafeImageChannel<T>) {
            unsafe {
                let ptr = std::ptr::slice_from_raw_parts(image.ptr, image.data as usize);
                Arc::<[T]>::from_raw(ptr);
            }
        }

        extern "C" fn clone_arc_channel<T: Clone>(
            image: &UnsafeImageChannel<T>,
        ) -> UnsafeImageChannel<T> {
            let arc = ManuallyDrop::new(unsafe {
                let ptr = std::ptr::slice_from_raw_parts(image.ptr, image.data as usize);
                Arc::<[T]>::from_raw(ptr)
            });
            UnsafeImageChannel::new_arc(
                (*arc).clone(),
                image.width,
                image.height,
                image.pixel_elements,
            )
        }

        &ImageChannelVTable {
            drop: clear_arc_channel,
            clone: clone_arc_channel,
            make_mut,
        }
    };
}

pub(crate) extern "C" fn clone_slice_into_arc_channel<T: Clone>(
    image: &UnsafeImageChannel<T>,
) -> UnsafeImageChannel<T> {
    UnsafeImageChannel::new_arc(
        Arc::from(image.buffer_flat()),
        image.width,
        image.height,
        image.pixel_elements,
    )
}
