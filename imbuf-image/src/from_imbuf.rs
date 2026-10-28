use std::num::{NonZeroU8, NonZeroUsize};

use image::ImageBuffer;
use imbuf::{
    Image, ImageChannel, IncompatibleChannelError, IncompatibleImage, IncompatibleImageError,
    PixelLayout, PixelType,
};

use crate::ref_image::{DynamicRefImage, DynamicRefImageMut};

/// Primitive sizes of the `image` formats. An `imbuf::DynamicImageChannel` only knows the size of
/// its primitive, so e.g. a primitive of 4 bytes is interpreted as `f32`.
const U8_SIZE: usize = size_of::<u8>();
const U16_SIZE: usize = size_of::<u16>();
const F32_SIZE: usize = size_of::<f32>();
/// Primitives of the `image` formats, reported as `target_pixel_type` of unsupported layouts
const SUPPORTED_PRIMITIVES: &str = "u8|u16|f32";

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
                = std::result::Result<DynamicRefImage<'a>, IncompatibleImageError>
            where
                Self: 'a;

            fn as_rust_image_dynamic(&self) -> Self::Result<'_> {
                match self {
                    $(image::DynamicImage::$variant(x) => Ok(x.into()),)+
                    _ => Err(unsupported_layout(color_layout(self.color()))),
                }
            }
        }

        impl ImbufAsImageMut for image::DynamicImage {
            type Result<'a>
                = std::result::Result<DynamicRefImageMut<'a>, IncompatibleImageError>
            where
                Self: 'a;

            fn as_rust_image_dynamic_mut(&mut self) -> Self::Result<'_> {
                match self {
                    $(image::DynamicImage::$variant(x) => Ok(x.into()),)+
                    _ => Err(unsupported_layout(color_layout(self.color()))),
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

impl<'a> TryFrom<&'a imbuf::DynamicImage> for DynamicRefImage<'a> {
    type Error = IncompatibleImageError;

    fn try_from(value: &'a imbuf::DynamicImage) -> Result<Self, Self::Error> {
        type DynRefImg<'a> = DynamicRefImage<'a>;
        value.ensure_single_channel()?;
        let channel = value.first();

        match (
            channel.bytes_per_primitive().get(),
            channel.pixel_elements().get(),
        ) {
            (U8_SIZE, 1) => Ok(DynRefImg::ImageLuma8(ref_image(channel.try_cast::<u8>()?))),
            (U8_SIZE, 2) => Ok(DynRefImg::ImageLumaA8(ref_image(
                channel.try_cast::<[u8; 2]>()?,
            ))),
            (U8_SIZE, 3) => Ok(DynRefImg::ImageRgb8(ref_image(
                channel.try_cast::<[u8; 3]>()?,
            ))),
            (U8_SIZE, 4) => Ok(DynRefImg::ImageRgba8(ref_image(
                channel.try_cast::<[u8; 4]>()?,
            ))),
            (U16_SIZE, 1) => Ok(DynRefImg::ImageLuma16(ref_image(
                channel.try_cast::<u16>()?,
            ))),
            (U16_SIZE, 2) => Ok(DynRefImg::ImageLumaA16(ref_image(
                channel.try_cast::<[u16; 2]>()?,
            ))),
            (U16_SIZE, 3) => Ok(DynRefImg::ImageRgb16(ref_image(
                channel.try_cast::<[u16; 3]>()?,
            ))),
            (U16_SIZE, 4) => Ok(DynRefImg::ImageRgba16(ref_image(
                channel.try_cast::<[u16; 4]>()?,
            ))),
            (F32_SIZE, 3) => Ok(DynRefImg::ImageRgb32F(ref_image(
                channel.try_cast::<[f32; 3]>()?,
            ))),
            (F32_SIZE, 4) => Ok(DynRefImg::ImageRgba32F(ref_image(
                channel.try_cast::<[f32; 4]>()?,
            ))),
            _ => Err(unsupported_layout(channel.pixel_layout())),
        }
    }
}

impl ImbufAsImage for imbuf::DynamicImage {
    type Result<'a>
        = std::result::Result<DynamicRefImage<'a>, IncompatibleImageError>
    where
        Self: 'a;

    fn as_rust_image_dynamic(&self) -> Self::Result<'_> {
        DynamicRefImage::try_from(self)
    }
}

impl ImbufAsImageMut for imbuf::DynamicImage {
    type Result<'a>
        = std::result::Result<DynamicRefImageMut<'a>, IncompatibleImageError>
    where
        Self: 'a;

    fn as_rust_image_dynamic_mut(&mut self) -> Self::Result<'_> {
        DynamicRefImageMut::try_from(self)
    }
}

impl<'a> TryFrom<&'a mut imbuf::DynamicImage> for DynamicRefImageMut<'a> {
    type Error = IncompatibleImageError;

    fn try_from(value: &'a mut imbuf::DynamicImage) -> Result<Self, Self::Error> {
        type DynRefImgMut<'a> = DynamicRefImageMut<'a>;
        value.ensure_single_channel()?;
        let channel = &mut value[0];

        match (
            channel.bytes_per_primitive().get(),
            channel.pixel_elements().get(),
        ) {
            (U8_SIZE, 1) => Ok(DynRefImgMut::ImageLuma8(ref_image_mut(
                channel.try_cast_mut::<u8>()?,
            ))),
            (U8_SIZE, 2) => Ok(DynRefImgMut::ImageLumaA8(ref_image_mut(
                channel.try_cast_mut::<[u8; 2]>()?,
            ))),
            (U8_SIZE, 3) => Ok(DynRefImgMut::ImageRgb8(ref_image_mut(
                channel.try_cast_mut::<[u8; 3]>()?,
            ))),
            (U8_SIZE, 4) => Ok(DynRefImgMut::ImageRgba8(ref_image_mut(
                channel.try_cast_mut::<[u8; 4]>()?,
            ))),
            (U16_SIZE, 1) => Ok(DynRefImgMut::ImageLuma16(ref_image_mut(
                channel.try_cast_mut::<u16>()?,
            ))),
            (U16_SIZE, 2) => Ok(DynRefImgMut::ImageLumaA16(ref_image_mut(
                channel.try_cast_mut::<[u16; 2]>()?,
            ))),
            (U16_SIZE, 3) => Ok(DynRefImgMut::ImageRgb16(ref_image_mut(
                channel.try_cast_mut::<[u16; 3]>()?,
            ))),
            (U16_SIZE, 4) => Ok(DynRefImgMut::ImageRgba16(ref_image_mut(
                channel.try_cast_mut::<[u16; 4]>()?,
            ))),
            (F32_SIZE, 3) => Ok(DynRefImgMut::ImageRgb32F(ref_image_mut(
                channel.try_cast_mut::<[f32; 3]>()?,
            ))),
            (F32_SIZE, 4) => Ok(DynRefImgMut::ImageRgba32F(ref_image_mut(
                channel.try_cast_mut::<[f32; 4]>()?,
            ))),
            _ => Err(unsupported_layout(channel.pixel_layout())),
        }
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
        impl ImbufAsImage for Image<$src, 1> {
            type Result<'a>
                = DynamicRefImage<'a>
            where
                Self: 'a;

            fn as_rust_image_dynamic(&self) -> Self::Result<'_> {
                DynamicRefImage::$variant(ref_image(self.first()))
            }
        }

        impl ImbufAsImageMut for Image<$src, 1> {
            type Result<'a>
                = DynamicRefImageMut<'a>
            where
                Self: 'a;

            fn as_rust_image_dynamic_mut(&mut self) -> Self::Result<'_> {
                DynamicRefImageMut::$variant(ref_image_mut(self.first_mut()))
            }
        }

        impl ImbufIntoImage for Image<$src, 1> {
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
    type Result = std::result::Result<image::DynamicImage, IncompatibleImageError>;

    fn into_rust_image_dynamic(self) -> Self::Result {
        fn convert<TP>(
            dynamic: imbuf::DynamicImage,
        ) -> Result<image::DynamicImage, IncompatibleImageError>
        where
            TP: PixelType,
            Image<TP, 1>: TryFrom<imbuf::DynamicImage, Error = IncompatibleImage<imbuf::DynamicImage>>
                + ImbufIntoImage<Result = image::DynamicImage>,
        {
            let image = Image::<TP, 1>::try_from(dynamic).map_err(|e| e.error)?;
            Ok(image.into_rust_image_dynamic())
        }

        self.ensure_single_channel()?;
        let channel = self.first();
        match (
            channel.bytes_per_primitive().get(),
            channel.pixel_elements().get(),
        ) {
            (U8_SIZE, 1) => convert::<u8>(self),
            (U8_SIZE, 2) => convert::<[u8; 2]>(self),
            (U8_SIZE, 3) => convert::<[u8; 3]>(self),
            (U8_SIZE, 4) => convert::<[u8; 4]>(self),
            (U16_SIZE, 1) => convert::<u16>(self),
            (U16_SIZE, 2) => convert::<[u16; 2]>(self),
            (U16_SIZE, 3) => convert::<[u16; 3]>(self),
            (U16_SIZE, 4) => convert::<[u16; 4]>(self),
            (F32_SIZE, 3) => convert::<[f32; 3]>(self),
            (F32_SIZE, 4) => convert::<[f32; 4]>(self),
            _ => Err(unsupported_layout(channel.pixel_layout())),
        }
    }
}

/// The conversion supports no pixel type with the `actual` layout
fn unsupported_layout(actual: PixelLayout) -> IncompatibleImageError {
    IncompatibleChannelError::new(actual, None, false, SUPPORTED_PRIMITIVES).into()
}

/// Layout of an `image` color type, which this crate doesn't know (`image::DynamicImage` is
/// `#[non_exhaustive]`)
fn color_layout(color: image::ColorType) -> PixelLayout {
    // A color has at least one channel of at least one byte, so `MIN` is never used
    let pixel_elements = NonZeroU8::new(color.channel_count()).unwrap_or(NonZeroU8::MIN);
    // `image` has no color with mixed primitive sizes
    let bytes_per_primitive =
        NonZeroU8::new(color.bytes_per_pixel() / pixel_elements).unwrap_or(NonZeroU8::MIN);
    PixelLayout::new(pixel_elements, NonZeroUsize::from(bytes_per_primitive))
}

fn image_from_raw<P, T>(width: u32, height: u32, buffer: Vec<T>) -> ImageBuffer<P, Vec<T>>
where
    P: image::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("Incompatible buffer size")
}

fn ref_image<P, TP>(channel: &ImageChannel<TP>) -> ImageBuffer<P, &[TP::Primitive]>
where
    P: image::Pixel<Subpixel = TP::Primitive>,
    TP: PixelType,
{
    let (width, height) = channel.dimensions();
    ImageBuffer::from_raw(width.get(), height.get(), channel.buffer_flat())
        .expect("Incompatible buffer size")
}

fn ref_image_mut<P, TP>(channel: &mut ImageChannel<TP>) -> ImageBuffer<P, &mut [TP::Primitive]>
where
    P: image::Pixel<Subpixel = TP::Primitive>,
    TP: PixelType,
{
    let (width, height) = channel.dimensions();
    ImageBuffer::from_raw(width.get(), height.get(), channel.make_mut_flat())
        .expect("Incompatible buffer size")
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::{io::Cursor, num::NonZeroU32};

    use imbuf::{Image, IncompatibleChannelCount};
    use testresult::TestResult;

    use super::*;

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_3: NonZeroU32 = NonZeroU32::new(3).unwrap();

    #[test]
    fn luma8_into_dynamic_and_back() -> TestResult {
        let mut expected = Cursor::new(Vec::new());
        let format = image::ImageFormat::Png;
        image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(1, 1, vec![1])
            .ok_or("buffer too small")?
            .write_to(&mut expected, format)?;
        let expected = expected.into_inner();
        let image = Image::<u8, 1>::new_vec(vec![1], NZ_1, NZ_1)?;

        let dynamic = imbuf::DynamicImage::from(image);
        assert_eq!(expected, test_encode(&dynamic.as_rust_image_dynamic()?)?);
        Ok(())
    }

    #[test]
    fn rgb16_into_dynamic_and_back() -> TestResult {
        let mut expected = Cursor::new(Vec::new());
        let format = image::ImageFormat::Png;
        image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(1, 1, vec![0, 1, 2])
            .ok_or("buffer too small")?
            .write_to(&mut expected, format)?;
        let expected = expected.into_inner();
        let image = Image::<[u16; 3], 1>::new_vec(vec![[0, 1, 2]], NZ_1, NZ_1)?;

        assert_eq!(expected, test_encode(&image.as_rust_image_dynamic())?);
        let dynamic = imbuf::DynamicImage::from(image);
        assert_eq!(expected, test_encode(&dynamic.as_rust_image_dynamic()?)?);
        Ok(())
    }

    #[test]
    fn into_image_round_trips() -> TestResult {
        let image = Image::<u8, 1>::new_vec(vec![1], NZ_1, NZ_1)?;
        assert_eq!(
            test_encode(&image.as_rust_image_dynamic())?,
            test_encode_owned(&image.into_rust_image_dynamic())?
        );
        Ok(())
    }

    #[test]
    fn dynamic_into_rust_image_returns_result() -> TestResult {
        let image = Image::<[u8; 3], 1>::new_vec(vec![[1, 2, 3]], NZ_1, NZ_1)?;
        let dynamic = imbuf::DynamicImage::from(image);

        let image::DynamicImage::ImageRgb8(buffer) = dynamic.into_rust_image_dynamic()? else {
            panic!("Expected Rgb8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        Ok(())
    }

    #[test]
    fn dynamic_as_rust_image_mut_writes_through() -> TestResult {
        let image = Image::<[f32; 3], 1>::new_vec(vec![[1., 2., 3.]], NZ_1, NZ_1)?;
        let mut dynamic = imbuf::DynamicImage::from(image);

        {
            let DynamicRefImageMut::ImageRgb32F(mut buffer) =
                dynamic.as_rust_image_dynamic_mut()?
            else {
                panic!("Expected Rgb32F");
            };
            buffer.as_mut()[0] = 4.;
        }

        assert_eq!(dynamic.first().buffer_flat::<f32>(), Ok(&[4., 2., 3.][..]));
        Ok(())
    }

    #[test]
    fn planar_dynamic_image_is_rejected() -> TestResult {
        let image = Image::<u8, 3>::new_vec(vec![1, 2, 3], NZ_1, NZ_1)?;
        let mut dynamic = imbuf::DynamicImage::from(image);

        let expected = IncompatibleImageError::from(
            dynamic
                .ensure_single_channel()
                .err()
                .ok_or("is single channel")?,
        );
        assert_matches!(
            &expected,
            IncompatibleImageError::ChannelCount(IncompatibleChannelCount { expected: NonZeroU8::MIN, actual, .. }) if actual.get() == 3
        );
        assert_eq!(
            dynamic.as_rust_image_dynamic().err().as_ref(),
            Some(&expected)
        );
        assert_eq!(
            dynamic.as_rust_image_dynamic_mut().err().as_ref(),
            Some(&expected)
        );
        assert_eq!(dynamic.into_rust_image_dynamic().err(), Some(expected));
        Ok(())
    }

    #[test]
    fn unsupported_layout_is_rejected() -> TestResult {
        // `image` has no `f32` luma image
        let image = Image::<f32, 1>::new_vec(vec![1.], NZ_1, NZ_1)?;
        let mut dynamic = imbuf::DynamicImage::from(image);

        let expected = IncompatibleImageError::Channel(IncompatibleChannelError::new(
            PixelLayout::new(NonZeroU8::MIN, NonZeroUsize::new(4).ok_or("zero")?),
            None,
            false,
            SUPPORTED_PRIMITIVES,
        ));
        assert_eq!(
            dynamic.as_rust_image_dynamic().err().as_ref(),
            Some(&expected)
        );
        assert_eq!(
            dynamic.as_rust_image_dynamic_mut().err().as_ref(),
            Some(&expected.clone())
        );
        assert_eq!(dynamic.into_rust_image_dynamic().err(), Some(expected));
        Ok(())
    }

    #[test]
    fn into_rust_image_shares_buffer() -> TestResult {
        let img = Image::<u8, 1>::new_vec(vec![1, 2, 3], NZ_3, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_rust_image_dynamic();
        let DynamicRefImage::ImageLuma8(buffer) = borrowed else {
            panic!("Expected Luma8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn into_rust_image_mut_shares_buffer() -> TestResult {
        let mut img = Image::<u8, 1>::new_vec(vec![1, 2, 3], NZ_3, NZ_1)?;
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
        Ok(())
    }

    #[test]
    fn as_rust_image_shares_buffer() -> TestResult {
        let img = Image::<u8, 1>::new_vec(vec![1, 2, 3], NZ_3, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_rust_image_dynamic();
        let DynamicRefImage::ImageLuma8(buffer) = borrowed else {
            panic!("Expected Luma8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn as_rust_image_works_through_mut_receiver() -> TestResult {
        let mut img = Image::<u8, 1>::new_vec(vec![1, 2, 3], NZ_3, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        let img_ref = &mut img;
        let borrowed = img_ref.as_rust_image_dynamic();
        let DynamicRefImage::ImageLuma8(buffer) = borrowed else {
            panic!("Expected Luma8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn as_rust_image_mut_shares_buffer() -> TestResult {
        let mut img = Image::<u8, 1>::new_vec(vec![1, 2, 3], NZ_3, NZ_1)?;
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
        Ok(())
    }

    #[test]
    fn rust_image_as_rust_image_shares_buffer() -> TestResult {
        let dynamic = image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).ok_or("buffer too small")?,
        );
        let data_ptr = dynamic.as_rgb8().ok_or("not rgb8")?.as_raw().as_ptr() as usize;

        let borrowed = dynamic.as_rust_image_dynamic()?;
        let DynamicRefImage::ImageRgb8(buffer) = borrowed else {
            panic!("Expected Rgb8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn rust_image_as_rust_image_mut_shares_buffer() -> TestResult {
        let mut dynamic = image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).ok_or("buffer too small")?,
        );
        let data_ptr = dynamic.as_rgb8().ok_or("not rgb8")?.as_raw().as_ptr() as usize;

        {
            let borrowed = dynamic.as_rust_image_dynamic_mut()?;
            let DynamicRefImageMut::ImageRgb8(mut buffer) = borrowed else {
                panic!("Expected Rgb8");
            };
            buffer.as_mut()[1] = 42;
        }

        assert_eq!(
            dynamic.as_rgb8().ok_or("not rgb8")?.as_raw(),
            &[1, 42, 3, 4, 5, 6]
        );
        assert_eq!(
            dynamic.as_rgb8().ok_or("not rgb8")?.as_raw().as_ptr() as usize,
            data_ptr
        );
        Ok(())
    }

    fn test_encode_owned(image: &image::DynamicImage) -> TestResult<Vec<u8>> {
        let mut buffer = Cursor::new(Vec::new());
        image.write_to(&mut buffer, image::ImageFormat::Png)?;
        Ok(buffer.into_inner())
    }

    fn test_encode(image: &DynamicRefImage<'_>) -> TestResult<Vec<u8>> {
        let mut expected = Cursor::new(Vec::new());
        let format = image::ImageFormat::Png;
        image.write_to(&mut expected, format)?;
        Ok(expected.into_inner())
    }
}
