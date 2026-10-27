use std::{
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32},
};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, PixelTypePrimitive,
    channel::{
        ChannelFactory, ImageChannelVTable, UnsafeImageChannel, calc_pixel_len_flat, primitive_size,
    },
};

struct VecFactory;

impl DynamicImageChannel {
    /// In contrast to the public constructors, `T` doesn't have to be `Pod`. Don't expose the
    /// result as `DynamicImageChannel`, unless `T` is `Pod`
    ///
    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height and `pixel_elements`.
    pub(crate) fn from_vec<T: PixelTypePrimitive>(
        input: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Self {
        assert_eq!(
            input.len(),
            calc_pixel_len_flat(width, height, pixel_elements),
            "Incompatible Buffer-Size"
        );
        let mut input = ManuallyDrop::new(input);
        let cap = input.capacity();
        let ptr = input.as_mut_ptr();

        let unsafe_ch = UnsafeImageChannel::new_with_vtable(
            ptr.cast(),
            width,
            height,
            pixel_elements,
            <VecFactory as ChannelFactory<T>>::VTABLE,
            std::ptr::without_provenance_mut(cap),
        );
        unsafe { DynamicImageChannel::from_unsafe(unsafe_ch) }
    }
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for VecFactory {
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable {
        make_mut,
        drop: drop_vec_channel::<T>,
        clone: crate::arc::clone_slice_into_arc_channel::<T>,
        primitive_size: primitive_size::<T>,
    };
}

unsafe extern "C-unwind" fn make_mut(_image: &mut DynamicImageChannelView<'_>) {
    // Do nothing, as the Vec is never shared (clone copies it)
}

pub(crate) unsafe extern "C-unwind" fn drop_vec_channel<T>(
    image: &mut DynamicImageChannelView<'_>,
) {
    let raw = DynamicImageChannelView::get_unsafe(image);
    drop(unsafe {
        Vec::from_raw_parts(
            raw.ptr.cast::<T>().cast_mut(),
            raw.calc_len_flat(),
            raw.data as usize,
        )
    });
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

        let planar = Image::<u8, 3>::from_flat_interleaved(&data, width, height);
        let interleaved_direct = Image::<[u8; 3], 1>::new_vec_flat(data, width, height);
        let interleaved_from_planar = Image::<[u8; 3], 1>::from_planar_image(&planar);

        assert_eq!(
            interleaved_direct.buffer(),
            interleaved_from_planar.buffer()
        );
    }

    #[test]
    fn new_vec_flat_with_capacity_which_is_no_multiple_of_pixel_elements() {
        let mut data = Vec::with_capacity(4);
        data.extend_from_slice(&[1u8, 2, 3]);
        let image = Image::<[u8; 3], 1>::new_vec_flat(data, NonZeroU32::MIN, NonZeroU32::MIN);
        assert_eq!(image.buffer(), &[[1, 2, 3]]);
    }
}
