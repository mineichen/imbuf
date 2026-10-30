use std::{
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32},
};

use bytemuck::Pod;

use crate::{
    DynamicImageChannel, DynamicImageChannelView, Image, ImageChannel, InvalidLenError, PixelType,
    PixelTypePrimitive,
    channel::{ChannelFactory, ImageChannelVTable, bytes_per_primitive},
};

struct VecFactory;

impl<const CHANNELS: usize, T: PixelType> Image<T, CHANNELS> {
    pub fn new_vec_flat(
        mut input: Vec<T::Primitive>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError>
    where
        T: PixelType,
    {
        Self::ensure_valid_flat_len(width, height, input.len())?;
        if CHANNELS > 1 {
            // The channels share the planar buffer, so it doesn't have to be cast to `T`
            // Safety: The length is checked above
            let channels = unsafe {
                crate::shared_vec::create_shared_channels_unchecked(
                    input,
                    [(width, height); CHANNELS],
                )
            };
            return Ok(Self::from_channels_unchecked(channels));
        }
        let pixel_elements = T::ELEMENTS.get() as usize;
        if input.capacity() % pixel_elements != 0 {
            // Vec<T> cannot represent the allocation. Boxed slices have an exact capacity.
            input = input.into_boxed_slice().into_vec();
        }

        let cap = input.capacity() / pixel_elements;
        let len = input.len() / pixel_elements;
        let ptr = input.as_mut_ptr().cast::<T>();
        std::mem::forget(input);
        let cast_input = unsafe { Vec::from_raw_parts(ptr, len, cap) };

        Self::new_vec(cast_input, width, height)
    }

    pub fn new_vec(
        input: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError>
    where
        T: Clone,
    {
        Self::ensure_valid_len(width, height, input.len())?;

        if CHANNELS == 1 {
            let channel = ImageChannel::<T>::new_vec(input, width, height)?;
            unsafe {
                let mut arr = std::mem::MaybeUninit::<[ImageChannel<T>; CHANNELS]>::uninit();
                std::ptr::write(arr.as_mut_ptr().cast::<ImageChannel<T>>(), channel);
                Ok(Self::from_channels_unchecked(arr.assume_init()))
            }
        } else {
            let flat_input = crate::pixel::flat_vec(input);
            // Safety: The length is checked above to be `width * height * CHANNELS`, so the flat
            // length is `width * height * T::ELEMENTS * CHANNELS`, which fits `usize`, as the
            // buffer exists
            let channels = unsafe {
                crate::shared_vec::create_shared_channels_unchecked(
                    flat_input,
                    [(width, height); CHANNELS],
                )
            };
            Ok(Self::from_channels_unchecked(channels))
        }
    }
}

impl<TP: PixelType> ImageChannel<TP> {
    pub fn new_vec(
        input: Vec<TP>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        DynamicImageChannelView::ensure_valid_len(width, height, input.len())?;

        let flat_input = crate::pixel::flat_vec(input);

        // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
        unsafe {
            let channel = DynamicImageChannel::new_vec_flat_unchecked(
                flat_input,
                width,
                height,
                TP::ELEMENTS,
            );

            Ok(Self::from_dynamic_unchecked(channel))
        }
    }
}

impl DynamicImageChannel {
    pub fn new_vec<T: PixelType + Pod>(
        data: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        DynamicImageChannel::ensure_valid_len(width, height, data.len())?;
        let data_flat = crate::pixel::flat_vec(data);
        // Safety: The length is checked above and T is Pod
        Ok(unsafe { Self::new_vec_flat_unchecked(data_flat, width, height, T::ELEMENTS) })
    }

    /// Creates a channel from a flat buffer with `pixel_elements` primitives per pixel, e.g. if
    /// `pixel_elements` is only known at runtime. Use [`DynamicImageChannel::new_vec`] for typed
    /// pixels (e.g. `Vec<[u8; 3]>`).
    ///
    /// Fails, if `width * height * pixel_elements` doesn't match `data.len()` or overflows `usize`.
    pub fn new_vec_flat<T: PixelTypePrimitive + Pod>(
        data: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Result<Self, InvalidLenError> {
        DynamicImageChannel::ensure_valid_flat_len(width, height, pixel_elements, data.len())?;
        // Safety: The length is checked above and T is Pod
        Ok(unsafe { Self::new_vec_flat_unchecked(data, width, height, pixel_elements) })
    }

    /// # Safety
    /// In contrast to the public constructors, `T` doesn't have to be `Pod`.
    /// This is only ok, if the `DynamicImageChannel` lives inside a `ImageChannel`, where it cannot be extracted.
    /// `DynamicImageChannel` themself may never have !Pod primitives, because upcasting just occurs based on size/alignment
    /// You could easily transform `Image<NonZeroU8>` into a `DynamicImage` and turn it into `Image<u8>` and modify it.
    /// The drop function, which assumes `NonZeroU8` could get Zero values, which is UB.
    /// Furthermore, `width`/`height`/`pixel_elements` are not checked against `input.len()` or for overflows
    #[must_use]
    pub unsafe fn new_vec_flat_unchecked<T: PixelTypePrimitive>(
        input: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Self {
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
        bytes_per_primitive: bytes_per_primitive::<T>,
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
    use assert_matches::assert_matches;
    use std::num::{NonZeroU8, NonZeroU32};

    use testresult::TestResult;

    use crate::{DynamicImageChannel, Image, InvalidLenError};

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    const NZ_3: NonZeroU32 = NonZeroU32::new(3).unwrap();
    const NZ_U8_3: NonZeroU8 = NonZeroU8::new(3).unwrap();

    #[test]
    fn create_interleaved_from_flat_vec() -> TestResult {
        let image: Image<[u8; 3], 1> = Image::new_vec(
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            NZ_2,
            NZ_2,
        )?;
        let buffer = image.buffer_flat().to_vec();
        assert_eq!(buffer.len(), 12);
        let image_from_flat: Image<[u8; 3], 1> = Image::new_vec_flat(buffer, NZ_2, NZ_2)?;

        assert_eq!(image, image_from_flat);
        Ok(())
    }

    #[test]
    fn new_vec_flat_reports_len_in_primitives() {
        // 3 primitives are a whole pixel, but 2 pixels are required
        assert_matches!(
            Image::<[u8; 3], 1>::new_vec_flat(vec![1, 2, 3], NZ_2, NZ_1),
            Err(InvalidLenError::BufferMismatch {
                expected: 6,
                actual: 3
            })
        );
    }

    #[test]
    fn new_vec_flat_with_runtime_pixel_elements() -> TestResult {
        let channel =
            DynamicImageChannel::new_vec_flat(vec![1u8, 2, 3, 4, 5, 6], NZ_2, NZ_1, NZ_U8_3)?;
        assert_eq!(NZ_U8_3, channel.pixel_elements());
        assert_eq!(&[[1u8, 2, 3], [4, 5, 6]], channel.buffer::<[u8; 3]>()?);
        Ok(())
    }

    #[test]
    fn from_vec_flat_with_wrong_len_fails() {
        let result = DynamicImageChannel::new_vec_flat(vec![1u8, 2, 3, 4, 5], NZ_2, NZ_1, NZ_U8_3);
        assert_matches!(
            result,
            Err(InvalidLenError::BufferMismatch {
                expected: 6,
                actual: 5
            })
        );
    }

    #[test]
    fn from_flat_vec_behaves_similar_to_from_interleaved() -> TestResult {
        let data = (0u8..18).collect::<Vec<_>>();

        let planar = Image::<u8, 3>::from_flat_interleaved(&data, NZ_2, NZ_3)?;
        let interleaved_direct = Image::<[u8; 3], 1>::new_vec_flat(data, NZ_2, NZ_3)?;
        let interleaved_from_planar = Image::<[u8; 3], 1>::from_planar_image(&planar);

        assert_eq!(
            interleaved_direct.buffer(),
            interleaved_from_planar.buffer()
        );
        Ok(())
    }

    #[test]
    fn new_vec_flat_with_capacity_which_is_no_multiple_of_pixel_elements() -> TestResult {
        let mut data = Vec::with_capacity(4);
        data.extend_from_slice(&[1u8, 2, 3]);
        let image = Image::<[u8; 3], 1>::new_vec_flat(data, NZ_1, NZ_1)?;
        assert_eq!(image.buffer(), &[[1, 2, 3]]);
        Ok(())
    }

    #[test]
    fn new_vec_flat_with_len_which_is_no_multiple_of_pixel_elements_fails() {
        let result = Image::<[u8; 3], 1>::new_vec_flat(vec![1, 2], NZ_1, NZ_1);
        assert_matches!(
            result,
            Err(crate::InvalidLenError::BufferMismatch {
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn new_vec_with_wrong_len_fails() {
        let result = Image::<u8, 2>::new_vec(vec![1, 2, 3], NZ_1, NZ_1);
        assert_matches!(
            result,
            Err(crate::InvalidLenError::BufferMismatch {
                expected: 2,
                actual: 3
            })
        );
    }
}
