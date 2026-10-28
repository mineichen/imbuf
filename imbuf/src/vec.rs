use std::{
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32},
};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, PixelTypePrimitive,
    channel::{ChannelFactory, ImageChannelVTable, primitive_size},
};

struct VecFactory;

impl DynamicImageChannel {
    /// #Safety
    /// In contrast to the public constructors, `T` doesn't have to be `Pod`.
    /// This is only ok, if the DynamicImageChannel lives inside a ImageChannel, where it cannot be extracted.
    /// `DynamicImageChannel` themself may never have !Pod primitives, because upcasting just occurs based on size/alignment
    /// You could easily transform Image<NonZeroU8> into a DynamicImage and turn it into `Image<u8>` and modify it.
    /// The drop function, which assumes `NonZeroU8` could get Zero values, which is UB.
    /// Furthermore, width/height/pixel_elements are not checked against input.len() or for overflows
    #[must_use]
    pub unsafe fn from_vec_flat_unchecked<T: PixelTypePrimitive>(
        input: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Self {
        // Panic, if the multiplication of the dimensions overflows or the length mismatches,
        // instead of comparing against a wrapped value
        debug_assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(
                width,
                height,
                pixel_elements,
                input.len()
            ),
            Ok(()),
            "Incompatible Buffer-Size"
        );
        let mut input = ManuallyDrop::new(input);
        let cap = input.capacity();
        let ptr = input.as_mut_ptr();

        // Safety: The buffer of the Vec stays valid until the vtable drops it and the vtable is
        // built for `T`
        unsafe {
            DynamicImageChannel::from_raw_parts(
                ptr.cast(),
                width,
                height,
                pixel_elements,
                <VecFactory as ChannelFactory<T>>::VTABLE,
                std::ptr::without_provenance_mut(cap),
            )
        }
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
    let ptr = DynamicImageChannelView::raw_ptr(image);
    let capacity = DynamicImageChannelView::raw_data(image) as usize;
    drop(unsafe { Vec::from_raw_parts(ptr.cast::<T>().cast_mut(), image.len_flat(), capacity) });
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
