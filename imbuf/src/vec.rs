use std::num::{NonZeroU8, NonZeroU32};

use crate::channel::{ChannelFactory, ImageChannelVTable, UnsafeImageChannel, calc_pixel_len_flat};

struct VecFactory;

impl<T: 'static> UnsafeImageChannel<T> {
    #[must_use]
    /// # Panics
    /// Panics if the buffer size is not compatible with the width and height.
    pub fn new_vec(
        input: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Self
    where
        T: Clone,
    {
        let cap = input.capacity();
        let ptr = input.as_ptr();

        assert_eq!(
            input.len(),
            calc_pixel_len_flat(width, height, pixel_elements),
            "Incompatible Buffer-Size"
        );
        std::mem::forget(input);
        let vtable = <VecFactory as ChannelFactory<T>>::VTABLE;
        unsafe {
            Self::new_with_vtable(
                ptr,
                width,
                height,
                pixel_elements,
                vtable,
                std::ptr::without_provenance_mut(cap),
            )
        }
    }
}

impl<T: 'static + Clone> ChannelFactory<T> for VecFactory {
    const VTABLE: &'static ImageChannelVTable<T> = {
        unsafe extern "C" fn make_mut<T: Clone>(_image: &mut UnsafeImageChannel<T>) {
            // Do nothing, as ptr is exclusive if i have &mut ImutableVTable
        }
        &ImageChannelVTable {
            make_mut,
            drop: clear_vec_channel,
            clone: crate::arc::clone_slice_into_arc_channel,
        }
    };
}

pub(crate) extern "C" fn clear_vec_channel<T>(image: &mut UnsafeImageChannel<T>) {
    unsafe {
        Vec::from_raw_parts(
            image.ptr.cast_mut(),
            (image.width.get() * image.height.get()) as usize * image.pixel_elements.get() as usize,
            image.data as usize,
        )
    };
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::Image;

    #[test]
    fn from_flat_vec_behaves_similar_to_from_interleaved() {
        let data = (0u8..18).collect::<Vec<_>>();
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(3).unwrap();

        let planar = Image::<u8, 3>::from_flat_interleaved(&data, (width, height));
        let interleaved_direct = Image::<[u8; 3], 1>::new_vec_flat(data, width, height);
        let interleaved_from_planar = Image::<[u8; 3], 1>::from_planar_image(&planar);

        assert_eq!(
            interleaved_direct.buffer(),
            interleaved_from_planar.buffer()
        );
    }
}
