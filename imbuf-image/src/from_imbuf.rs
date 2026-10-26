use std::num::NonZeroU8;

use image::ImageBuffer;
use imbuf::{GenericDynamicImage, GenericImage, GenericImageChannel, PixelType};

use crate::{
    non_zero_u8,
    ref_image::{DynamicRefImage, DynamicRefImageMut},
};

/// Primitive sizes of the `image` formats. An `imbuf::DynamicImageChannel` only knows the size of
/// its primitive, so e.g. a primitive of 4 bytes is interpreted as `f32`.
const U8_SIZE: usize = size_of::<u8>();
const U16_SIZE: usize = size_of::<u16>();
const F32_SIZE: usize = size_of::<f32>();

#[derive(Debug, thiserror::Error)]
#[error(
    "Incompatible DynamicImage: channels={channels:?}(must always be 1), pixel_elements={pixel_elements:?}"
)]
pub struct IncompatibleLayoutError {
    channels: Option<NonZeroU8>,
    pixel_elements: Option<NonZeroU8>,
}

/// Borrows an [`imbuf`] buffer as an [`image`] value.
pub trait ImbufAsImage {
    /// The returned image for infallible conversions, or a `Result` for
    /// fallible ones.
    type Result<'a>
    where
        Self: 'a;

    /// Borrows the buffer as a shared [`DynamicRefImage`] without copying.
    fn as_rust_image_dynamic(&self) -> Self::Result<'_>;
}

/// Mutably borrows an [`imbuf`] buffer as an [`image`] value.
pub trait ImbufAsImageMut {
    /// The returned image for infallible conversions, or a `Result` for
    /// fallible ones.
    type Result<'a>
    where
        Self: 'a;

    /// Borrows the buffer as a [`DynamicRefImageMut`] without copying.
    fn as_rust_image_dynamic_mut(&mut self) -> Self::Result<'_>;
}

macro_rules! impl_from_borrowed_image_buffer {
    ($($variant:ident),+ $(,)?) => {
        impl ImbufAsImage for image::DynamicImage {
            type Result<'a>
                = std::result::Result<DynamicRefImage<'a>, IncompatibleLayoutError>
            where
                Self: 'a;

            fn as_rust_image_dynamic(&self) -> Self::Result<'_> {
                match self {
                    $(image::DynamicImage::$variant(x) => Ok(x.into()),)+
                    _ => Err(IncompatibleLayoutError {
                        channels: None,
                        pixel_elements: None,
                    }),
                }
            }
        }

        impl ImbufAsImageMut for image::DynamicImage {
            type Result<'a>
                = std::result::Result<DynamicRefImageMut<'a>, IncompatibleLayoutError>
            where
                Self: 'a;

            fn as_rust_image_dynamic_mut(&mut self) -> Self::Result<'_> {
                match self {
                    $(image::DynamicImage::$variant(x) => Ok(x.into()),)+
                    _ => Err(IncompatibleLayoutError {
                        channels: None,
                        pixel_elements: None,
                    }),
                }
            }
        }
    };
}

impl_from_borrowed_image_buffer!(
    ImageLuma8,
    ImageLumaA8,
    ImageRgb8,
    ImageRgba8,
    ImageLuma16,
    ImageLumaA16,
    ImageRgb16,
    ImageRgba16,
    ImageRgb32F,
    ImageRgba32F,
);

impl<'a> TryFrom<&'a GenericDynamicImage<'_>> for DynamicRefImage<'a> {
    type Error = IncompatibleLayoutError;

    fn try_from(value: &'a GenericDynamicImage<'_>) -> Result<Self, Self::Error> {
        let channel = value.first();
        let pixel_elements = channel.pixel_elements();
        let error = IncompatibleLayoutError {
            channels: Some(non_zero_u8(value.len_nonzero().get())),
            pixel_elements: Some(pixel_elements),
        };

        if value.len() != 1 {
            return Err(error);
        }

        type DynRefImg<'a> = DynamicRefImage<'a>;
        match (channel.primitive_size().get(), pixel_elements.get()) {
            (U8_SIZE, 1) => channel
                .try_cast::<u8>()
                .map(|x| DynRefImg::ImageLuma8(ref_image(x))),
            (U8_SIZE, 2) => channel
                .try_cast::<[u8; 2]>()
                .map(|x| DynRefImg::ImageLumaA8(ref_image(x))),
            (U8_SIZE, 3) => channel
                .try_cast::<[u8; 3]>()
                .map(|x| DynRefImg::ImageRgb8(ref_image(x))),
            (U8_SIZE, 4) => channel
                .try_cast::<[u8; 4]>()
                .map(|x| DynRefImg::ImageRgba8(ref_image(x))),
            (U16_SIZE, 1) => channel
                .try_cast::<u16>()
                .map(|x| DynRefImg::ImageLuma16(ref_image(x))),
            (U16_SIZE, 2) => channel
                .try_cast::<[u16; 2]>()
                .map(|x| DynRefImg::ImageLumaA16(ref_image(x))),
            (U16_SIZE, 3) => channel
                .try_cast::<[u16; 3]>()
                .map(|x| DynRefImg::ImageRgb16(ref_image(x))),
            (U16_SIZE, 4) => channel
                .try_cast::<[u16; 4]>()
                .map(|x| DynRefImg::ImageRgba16(ref_image(x))),
            (F32_SIZE, 3) => channel
                .try_cast::<[f32; 3]>()
                .map(|x| DynRefImg::ImageRgb32F(ref_image(x))),
            (F32_SIZE, 4) => channel
                .try_cast::<[f32; 4]>()
                .map(|x| DynRefImg::ImageRgba32F(ref_image(x))),
            _ => None,
        }
        .ok_or(error)
    }
}

impl ImbufAsImage for GenericDynamicImage<'_> {
    type Result<'a>
        = std::result::Result<DynamicRefImage<'a>, IncompatibleLayoutError>
    where
        Self: 'a;

    fn as_rust_image_dynamic(&self) -> Self::Result<'_> {
        DynamicRefImage::try_from(self)
    }
}

impl ImbufAsImageMut for GenericDynamicImage<'_> {
    type Result<'a>
        = std::result::Result<DynamicRefImageMut<'a>, IncompatibleLayoutError>
    where
        Self: 'a;

    fn as_rust_image_dynamic_mut(&mut self) -> Self::Result<'_> {
        DynamicRefImageMut::try_from(self)
    }
}

impl<'a> TryFrom<&'a mut GenericDynamicImage<'_>> for DynamicRefImageMut<'a> {
    type Error = IncompatibleLayoutError;

    fn try_from(value: &'a mut GenericDynamicImage<'_>) -> Result<Self, Self::Error> {
        let pixel_elements = value.first().pixel_elements();
        let error = IncompatibleLayoutError {
            channels: Some(non_zero_u8(value.len())),
            pixel_elements: Some(pixel_elements),
        };
        if value.len() != 1 {
            return Err(error);
        }
        let channel = &mut value[0];
        type DynRefImgMut<'a> = DynamicRefImageMut<'a>;
        match (channel.primitive_size().get(), pixel_elements.get()) {
            (U8_SIZE, 1) => channel
                .try_cast_mut::<u8>()
                .map(|x| DynRefImgMut::ImageLuma8(ref_image_mut(x))),
            (U8_SIZE, 2) => channel
                .try_cast_mut::<[u8; 2]>()
                .map(|x| DynRefImgMut::ImageLumaA8(ref_image_mut(x))),
            (U8_SIZE, 3) => channel
                .try_cast_mut::<[u8; 3]>()
                .map(|x| DynRefImgMut::ImageRgb8(ref_image_mut(x))),
            (U8_SIZE, 4) => channel
                .try_cast_mut::<[u8; 4]>()
                .map(|x| DynRefImgMut::ImageRgba8(ref_image_mut(x))),
            (U16_SIZE, 1) => channel
                .try_cast_mut::<u16>()
                .map(|x| DynRefImgMut::ImageLuma16(ref_image_mut(x))),
            (U16_SIZE, 2) => channel
                .try_cast_mut::<[u16; 2]>()
                .map(|x| DynRefImgMut::ImageLumaA16(ref_image_mut(x))),
            (U16_SIZE, 3) => channel
                .try_cast_mut::<[u16; 3]>()
                .map(|x| DynRefImgMut::ImageRgb16(ref_image_mut(x))),
            (U16_SIZE, 4) => channel
                .try_cast_mut::<[u16; 4]>()
                .map(|x| DynRefImgMut::ImageRgba16(ref_image_mut(x))),
            (F32_SIZE, 3) => channel
                .try_cast_mut::<[f32; 3]>()
                .map(|x| DynRefImgMut::ImageRgb32F(ref_image_mut(x))),
            (F32_SIZE, 4) => channel
                .try_cast_mut::<[f32; 4]>()
                .map(|x| DynRefImgMut::ImageRgba32F(ref_image_mut(x))),
            _ => None,
        }
        .ok_or(error)
    }
}

/// Converts an owned [`imbuf`] image into an owned [`image::DynamicImage`].
///
/// Uniquely owned buffers are moved without copying; shared buffers are copied.
/// The concrete return type is a direct image for infallible conversions or a
/// `Result` for fallible ones.
pub trait ImbufIntoImage {
    /// The returned image for infallible conversions, or a `Result` for
    /// fallible ones.
    type Result;

    fn into_rust_image_dynamic(self) -> Self::Result;
}

macro_rules! impl_imbuf_to_image {
    ($src:ty, $variant:ident) => {
        impl ImbufAsImage for GenericImage<'_, $src, 1> {
            type Result<'a>
                = DynamicRefImage<'a>
            where
                Self: 'a;

            fn as_rust_image_dynamic(&self) -> Self::Result<'_> {
                DynamicRefImage::$variant(ref_image(self.first()))
            }
        }

        impl ImbufAsImageMut for GenericImage<'_, $src, 1> {
            type Result<'a>
                = DynamicRefImageMut<'a>
            where
                Self: 'a;

            fn as_rust_image_dynamic_mut(&mut self) -> Self::Result<'_> {
                DynamicRefImageMut::$variant(ref_image_mut(self.first_mut()))
            }
        }

        impl ImbufIntoImage for GenericImage<'_, $src, 1> {
            type Result = image::DynamicImage;

            fn into_rust_image_dynamic(self) -> Self::Result {
                let (width, height) = self.dimensions();
                let [channel] = <[_; 1]>::from(self);
                let buffer = channel.into_vec_flat();
                image::DynamicImage::$variant(image_from_raw(width.get(), height.get(), buffer))
            }
        }
    };
}

impl_imbuf_to_image!(u8, ImageLuma8);
impl_imbuf_to_image!(u16, ImageLuma16);
impl_imbuf_to_image!([u8; 2], ImageLumaA8);
impl_imbuf_to_image!([u16; 2], ImageLumaA16);
impl_imbuf_to_image!([u8; 3], ImageRgb8);
impl_imbuf_to_image!([u16; 3], ImageRgb16);
impl_imbuf_to_image!([f32; 3], ImageRgb32F);
impl_imbuf_to_image!([u8; 4], ImageRgba8);
impl_imbuf_to_image!([u16; 4], ImageRgba16);
impl_imbuf_to_image!([f32; 4], ImageRgba32F);

impl ImbufIntoImage for imbuf::DynamicImage {
    type Result = std::result::Result<image::DynamicImage, IncompatibleLayoutError>;

    fn into_rust_image_dynamic(self) -> Self::Result {
        fn convert<TP, const N: usize>(
            dynamic: imbuf::DynamicImage,
            error: IncompatibleLayoutError,
        ) -> Result<image::DynamicImage, IncompatibleLayoutError>
        where
            TP: PixelType,
            imbuf::Image<TP, N>:
                TryFrom<imbuf::DynamicImage> + ImbufIntoImage<Result = image::DynamicImage>,
        {
            let image = imbuf::Image::<TP, N>::try_from(dynamic).map_err(|_| error)?;
            Ok(image.into_rust_image_dynamic())
        }

        let pixel_elements = self.first().pixel_elements();
        let error = IncompatibleLayoutError {
            channels: Some(non_zero_u8(self.len())),
            pixel_elements: Some(pixel_elements),
        };
        if self.len() != 1 {
            return Err(error);
        }

        match (self.first().primitive_size().get(), pixel_elements.get()) {
            (U8_SIZE, 1) => convert::<u8, 1>(self, error),
            (U8_SIZE, 2) => convert::<[u8; 2], 1>(self, error),
            (U8_SIZE, 3) => convert::<[u8; 3], 1>(self, error),
            (U8_SIZE, 4) => convert::<[u8; 4], 1>(self, error),
            (U16_SIZE, 1) => convert::<u16, 1>(self, error),
            (U16_SIZE, 2) => convert::<[u16; 2], 1>(self, error),
            (U16_SIZE, 3) => convert::<[u16; 3], 1>(self, error),
            (U16_SIZE, 4) => convert::<[u16; 4], 1>(self, error),
            (F32_SIZE, 3) => convert::<[f32; 3], 1>(self, error),
            (F32_SIZE, 4) => convert::<[f32; 4], 1>(self, error),
            _ => Err(error),
        }
    }
}

fn image_from_raw<P, T>(width: u32, height: u32, buffer: Vec<T>) -> ImageBuffer<P, Vec<T>>
where
    P: image::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("Incompatible buffer size")
}

fn ref_image<'a, P, TP>(
    channel: &'a GenericImageChannel<'_, TP>,
) -> ImageBuffer<P, &'a [TP::Primitive]>
where
    P: image::Pixel<Subpixel = TP::Primitive>,
    TP: PixelType,
{
    let (width, height) = channel.dimensions();
    ImageBuffer::from_raw(width.get(), height.get(), channel.buffer_flat())
        .expect("Incompatible buffer size")
}

fn ref_image_mut<'a, P, TP>(
    channel: &'a mut GenericImageChannel<'_, TP>,
) -> ImageBuffer<P, &'a mut [TP::Primitive]>
where
    P: image::Pixel<Subpixel = TP::Primitive>,
    TP: PixelType,
{
    let (width, height) = channel.dimensions();
    ImageBuffer::from_raw(width.get(), height.get(), channel.primitive_make_mut())
        .expect("Incompatible buffer size")
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, num::NonZeroU32};

    use imbuf::Image;

    use super::*;

    #[test]
    fn luma8_into_dynamic_and_back() {
        let mut expected = Cursor::new(Vec::new());
        let format = image::ImageFormat::Png;
        image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(1, 1, vec![1])
            .unwrap()
            .write_to(&mut expected, format)
            .unwrap();
        let expected = expected.into_inner();
        let image = Image::<u8, 1>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);

        let dynamic = imbuf::DynamicImage::from(image);
        assert_eq!(
            expected,
            test_encode(dynamic.as_rust_image_dynamic().unwrap())
        );
    }

    #[test]
    fn rgb16_into_dynamic_and_back() {
        let mut expected = Cursor::new(Vec::new());
        let format = image::ImageFormat::Png;
        image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(1, 1, vec![0, 1, 2])
            .unwrap()
            .write_to(&mut expected, format)
            .unwrap();
        let expected = expected.into_inner();
        let image =
            Image::<[u16; 3], 1>::new_vec(vec![[0, 1, 2]], NonZeroU32::MIN, NonZeroU32::MIN);

        assert_eq!(expected, test_encode(image.as_rust_image_dynamic()));
        let dynamic = imbuf::DynamicImage::from(image);
        assert_eq!(
            expected,
            test_encode(dynamic.as_rust_image_dynamic().unwrap())
        );
    }

    #[test]
    fn borrowed_imbuf_image_as_rust_image() {
        let mut data = [[1u8, 2, 3]];
        let data_ptr = data.as_ptr().cast::<u8>();
        let mut image = imbuf::ImageRef::<[u8; 3], 1>::new_slice_mut(
            &mut data,
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        );
        let image::DynamicImage::ImageRgb8(_) = image.view().into_rust_image_dynamic() else {
            panic!("Expected Rgb8");
        };
        let DynamicRefImageMut::ImageRgb8(mut buffer) = image.as_rust_image_dynamic_mut() else {
            panic!("Expected Rgb8");
        };
        assert_eq!(buffer.as_ptr(), data_ptr);
        buffer.get_pixel_mut(0, 0).0[0] = 4;

        let mut dynamic = imbuf::GenericDynamicImage::from(image);
        let DynamicRefImage::ImageRgb8(buffer) = dynamic.as_rust_image_dynamic().unwrap() else {
            panic!("Expected Rgb8");
        };
        assert_eq!(buffer.as_ptr(), data_ptr);
        let Ok(DynamicRefImageMut::ImageRgb8(mut buffer)) = dynamic.as_rust_image_dynamic_mut()
        else {
            panic!("Expected Rgb8");
        };
        buffer.get_pixel_mut(0, 0).0[1] = 5;
        drop(dynamic);
        assert_eq!(data, [[4, 5, 3]]);
    }

    #[test]
    fn into_image_round_trips() {
        let image = Image::<u8, 1>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        assert_eq!(
            test_encode(image.as_rust_image_dynamic()),
            test_encode_owned(image.into_rust_image_dynamic())
        );
    }

    #[test]
    fn dynamic_into_rust_image_returns_result() {
        let image = Image::<[u8; 3], 1>::new_vec(vec![[1, 2, 3]], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = imbuf::DynamicImage::from(image);

        let image::DynamicImage::ImageRgb8(buffer) = dynamic.into_rust_image_dynamic().unwrap()
        else {
            panic!("Expected Rgb8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
    }

    #[test]
    fn dynamic_as_rust_image_mut_writes_through() {
        let image =
            Image::<[f32; 3], 1>::new_vec(vec![[1., 2., 3.]], NonZeroU32::MIN, NonZeroU32::MIN);
        let mut dynamic = imbuf::DynamicImage::from(image);

        {
            let DynamicRefImageMut::ImageRgb32F(mut buffer) =
                dynamic.as_rust_image_dynamic_mut().unwrap()
            else {
                panic!("Expected Rgb32F");
            };
            buffer.as_mut()[0] = 4.;
        }

        assert_eq!(
            dynamic.first().buffer_flat::<f32>(),
            Some(&[4., 2., 3.][..])
        );
    }

    #[test]
    fn planar_dynamic_image_is_rejected() {
        let image = Image::<u8, 3>::new_vec(vec![1, 2, 3], NonZeroU32::MIN, NonZeroU32::MIN);
        let mut dynamic = imbuf::DynamicImage::from(image);

        dynamic.as_rust_image_dynamic().unwrap_err();
        dynamic.as_rust_image_dynamic_mut().unwrap_err();
        dynamic.into_rust_image_dynamic().unwrap_err();
    }

    #[test]
    fn into_rust_image_shares_buffer() {
        let img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_rust_image_dynamic();
        let DynamicRefImage::ImageLuma8(buffer) = borrowed else {
            panic!("Expected Luma8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn into_rust_image_mut_shares_buffer() {
        let mut img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let borrowed = img.as_rust_image_dynamic_mut();
            let DynamicRefImageMut::ImageLuma8(mut buffer) = borrowed else {
                panic!("Expected Luma8");
            };
            buffer.as_mut()[0] = 42;
        }

        assert_eq!(img.buffer(), &[42, 2, 3]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn as_rust_image_shares_buffer() {
        let img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_rust_image_dynamic();
        let DynamicRefImage::ImageLuma8(buffer) = borrowed else {
            panic!("Expected Luma8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn as_rust_image_works_through_mut_receiver() {
        let mut img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        let img_ref = &mut img;
        let borrowed = img_ref.as_rust_image_dynamic();
        let DynamicRefImage::ImageLuma8(buffer) = borrowed else {
            panic!("Expected Luma8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn as_rust_image_mut_shares_buffer() {
        let mut img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let borrowed = img.as_rust_image_dynamic_mut();
            let DynamicRefImageMut::ImageLuma8(mut buffer) = borrowed else {
                panic!("Expected Luma8");
            };
            buffer.as_mut()[0] = 42;
        }

        assert_eq!(img.buffer(), &[42, 2, 3]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn rust_image_as_rust_image_shares_buffer() {
        let dynamic = image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap(),
        );
        let data_ptr = dynamic.as_rgb8().unwrap().as_raw().as_ptr() as usize;

        let borrowed = dynamic.as_rust_image_dynamic().unwrap();
        let DynamicRefImage::ImageRgb8(buffer) = borrowed else {
            panic!("Expected Rgb8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn rust_image_as_rust_image_mut_shares_buffer() {
        let mut dynamic = image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap(),
        );
        let data_ptr = dynamic.as_rgb8().unwrap().as_raw().as_ptr() as usize;

        {
            let borrowed = dynamic.as_rust_image_dynamic_mut().unwrap();
            let DynamicRefImageMut::ImageRgb8(mut buffer) = borrowed else {
                panic!("Expected Rgb8");
            };
            buffer.as_mut()[1] = 42;
        }

        assert_eq!(dynamic.as_rgb8().unwrap().as_raw(), &[1, 42, 3, 4, 5, 6]);
        assert_eq!(
            dynamic.as_rgb8().unwrap().as_raw().as_ptr() as usize,
            data_ptr
        );
    }

    fn test_encode_owned(image: image::DynamicImage) -> Vec<u8> {
        let mut buffer = Cursor::new(Vec::new());
        image
            .write_to(&mut buffer, image::ImageFormat::Png)
            .unwrap();
        buffer.into_inner()
    }

    fn test_encode(image: DynamicRefImage<'_>) -> Vec<u8> {
        let mut expected = Cursor::new(Vec::new());
        let format = image::ImageFormat::Png;
        image.write_to(&mut expected, format).unwrap();
        expected.into_inner()
    }
}
