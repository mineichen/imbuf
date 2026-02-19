use std::num::{NonZeroU32, NonZeroU8};

use image_0_25::{DynamicImage, GenericImageView, ImageBuffer, Luma, LumaA, Rgb, Rgba};

use crate::{DynamicImageChannel, Image, IncompatibleBufferSize};

#[derive(thiserror::Error, Debug)]
#[error("Cannot convert {image:?} into DynamicImage: {reason}")]
#[non_exhaustive]
pub struct IntoDynamicImage0_25Error {
    pub image: DynamicImage,
    pub reason: IntoDynamicImage0_25ErrorReason,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IntoDynamicImage0_25ErrorReason {
    #[error("Neither height nor length can be 0")]
    ZeroDimension(#[from] std::num::TryFromIntError),
    #[error(transparent)]
    IncompatibleBufferSize(#[from] IncompatibleBufferSize),
    #[error(
        "::image::DynamicImage is non_exhaustive, so it could be extended in the future. Report this as a bug/extension request to imbuf"
    )]
    NonExhaustive,
}

#[derive(Debug)]
pub enum DynamicRefImage0_25<'a> {
    ImageLuma8(ImageBuffer<Luma<u8>, &'a [u8]>),
    ImageLuma16(ImageBuffer<Luma<u16>, &'a [u16]>),
    ImageLumaA8(ImageBuffer<LumaA<u8>, &'a [u8]>),
    ImageLumaA16(ImageBuffer<LumaA<u16>, &'a [u16]>),
    ImageRgb8(ImageBuffer<Rgb<u8>, &'a [u8]>),
    ImageRgb16(ImageBuffer<Rgb<u16>, &'a [u16]>),
    ImageRgb32F(ImageBuffer<Rgb<f32>, &'a [f32]>),
    ImageRgba8(ImageBuffer<Rgba<u8>, &'a [u8]>),
    ImageRgba16(ImageBuffer<Rgba<u16>, &'a [u16]>),
    ImageRgba32F(ImageBuffer<Rgba<f32>, &'a [f32]>),
}

#[derive(Debug, thiserror::Error)]
pub enum IntoDynamicRefImage0_25Error {
    #[error(
        "Incompatible DynamicImage: channels={channels}(must always be 1), pixel_elements={pixel_elements}"
    )]
    IncompatibleLayout {
        channels: NonZeroU8,
        pixel_elements: NonZeroU8,
    },
}

impl DynamicRefImage0_25<'_> {
    /// # Errors
    /// Forwards errors of `image::write_to`
    pub fn write_to<W: std::io::Write + std::io::Seek>(
        &self,
        mut buffer: W,
        format: image_0_25::ImageFormat,
    ) -> Result<(), image_0_25::ImageError> {
        match self {
            DynamicRefImage0_25::ImageLuma8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageLuma16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageLumaA8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageLumaA16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageRgb8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageRgb16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageRgb32F(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageRgba8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageRgba16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage0_25::ImageRgba32F(x) => x.write_to(&mut buffer, format),
        }
    }
}

impl<'a> TryFrom<&'a crate::DynamicImage> for DynamicRefImage0_25<'a> {
    type Error = IntoDynamicRefImage0_25Error;

    fn try_from(value: &'a crate::DynamicImage) -> Result<Self, Self::Error> {
        let channel = value.first();
        let pixel_elements = channel.pixel_elements();

        if value.len() != 1 {
            return Err(IntoDynamicRefImage0_25Error::IncompatibleLayout {
                channels: crate::unwrap_usize_to_nonzero_u8(value.len()),
                pixel_elements,
            });
        }

        let width = channel.width().get();
        let height = channel.height().get();

        match (channel, pixel_elements.get()) {
            (DynamicImageChannel::U8(x), 1) => {
                Ok(DynamicRefImage0_25::ImageLuma8(ref_image::<Luma<u8>, u8>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U8(x), 2) => {
                Ok(DynamicRefImage0_25::ImageLumaA8(
                    ref_image::<LumaA<u8>, u8>(width, height, x.buffer_flat()),
                ))
            }
            (DynamicImageChannel::U8(x), 3) => {
                Ok(DynamicRefImage0_25::ImageRgb8(ref_image::<Rgb<u8>, u8>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U8(x), 4) => {
                Ok(DynamicRefImage0_25::ImageRgba8(ref_image::<Rgba<u8>, u8>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U16(x), 1) => {
                Ok(DynamicRefImage0_25::ImageLuma16(
                    ref_image::<Luma<u16>, u16>(width, height, x.buffer_flat()),
                ))
            }
            (DynamicImageChannel::U16(x), 2) => Ok(DynamicRefImage0_25::ImageLumaA16(ref_image::<
                LumaA<u16>,
                u16,
            >(
                width,
                height,
                x.buffer_flat(),
            ))),
            (DynamicImageChannel::U16(x), 3) => {
                Ok(DynamicRefImage0_25::ImageRgb16(ref_image::<Rgb<u16>, u16>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U16(x), 4) => {
                Ok(DynamicRefImage0_25::ImageRgba16(
                    ref_image::<Rgba<u16>, u16>(width, height, x.buffer_flat()),
                ))
            }
            (DynamicImageChannel::F32(x), 3) => {
                Ok(DynamicRefImage0_25::ImageRgb32F(
                    ref_image::<Rgb<f32>, f32>(width, height, x.buffer_flat()),
                ))
            }
            (DynamicImageChannel::F32(x), 4) => {
                Ok(DynamicRefImage0_25::ImageRgba32F(
                    ref_image::<Rgba<f32>, f32>(width, height, x.buffer_flat()),
                ))
            }
            (_, actual) => Err(IntoDynamicRefImage0_25Error::IncompatibleLayout {
                channels: crate::unwrap_usize_to_nonzero_u8(value.len()),
                pixel_elements: NonZeroU8::new(actual).unwrap(),
            }),
        }
    }
}

macro_rules! impl_from_image_ref_dynamic {
    ($src:ty, $pixel:ty, $variant:ident) => {
        impl<'a> From<&'a Image<$src, 1>> for DynamicRefImage0_25<'a> {
            fn from(value: &'a Image<$src, 1>) -> Self {
                let (width, height) = value.dimensions();
                let buffer = value.buffer_flat();
                DynamicRefImage0_25::$variant(ref_image(width.get(), height.get(), buffer))
            }
        }
    };
}

impl_from_image_ref_dynamic!(u8, Luma<u8>, ImageLuma8);
impl_from_image_ref_dynamic!(u16, Luma<u16>, ImageLuma16);
impl_from_image_ref_dynamic!([u8; 2], LumaA<u8>, ImageLumaA8);
impl_from_image_ref_dynamic!([u16; 2], LumaA<u16>, ImageLumaA16);
impl_from_image_ref_dynamic!([u8; 3], Rgb<u8>, ImageRgb8);
impl_from_image_ref_dynamic!([u16; 3], Rgb<u16>, ImageRgb16);
impl_from_image_ref_dynamic!([f32; 3], Rgb<f32>, ImageRgb32F);
impl_from_image_ref_dynamic!([u8; 4], Rgba<u8>, ImageRgba8);
impl_from_image_ref_dynamic!([u16; 4], Rgba<u16>, ImageRgba16);
impl_from_image_ref_dynamic!([f32; 4], Rgba<f32>, ImageRgba32F);

macro_rules! impl_from_image_dynamic {
    ($src:ty, $pixel:ty, $variant:ident) => {
        impl From<Image<$src, 1>> for DynamicImage {
            fn from(value: Image<$src, 1>) -> Self {
                let (width, height) = value.dimensions();
                let [channel] = <[_; 1]>::from(value);
                let buffer = channel.into_vec_flat();
                DynamicImage::$variant(image_from_raw(width.get(), height.get(), buffer))
            }
        }
    };
}

impl_from_image_dynamic!(u8, Luma<u8>, ImageLuma8);
impl_from_image_dynamic!(u16, Luma<u16>, ImageLuma16);
impl_from_image_dynamic!([u8; 2], LumaA<u8>, ImageLumaA8);
impl_from_image_dynamic!([u16; 2], LumaA<u16>, ImageLumaA16);
impl_from_image_dynamic!([u8; 3], Rgb<u8>, ImageRgb8);
impl_from_image_dynamic!([u16; 3], Rgb<u16>, ImageRgb16);
impl_from_image_dynamic!([f32; 3], Rgb<f32>, ImageRgb32F);
impl_from_image_dynamic!([u8; 4], Rgba<u8>, ImageRgba8);
impl_from_image_dynamic!([u16; 4], Rgba<u16>, ImageRgba16);
impl_from_image_dynamic!([f32; 4], Rgba<f32>, ImageRgba32F);

/// Only fails, if `image::Image.width()` or `image::Image.height()` is 0
impl TryFrom<DynamicImage> for crate::DynamicImage {
    type Error = IntoDynamicImage0_25Error;

    fn try_from(value: DynamicImage) -> Result<Self, Self::Error> {
        let (width, height) = value.dimensions();
        let width_times_height = width as usize * height as usize;
        let width = match NonZeroU32::try_from(width) {
            Ok(width) => width,
            Err(e) => {
                return Err(IntoDynamicImage0_25Error {
                    image: value,
                    reason: e.into(),
                });
            }
        };

        let height = match NonZeroU32::try_from(height) {
            Ok(height) => height,
            Err(e) => {
                return Err(IntoDynamicImage0_25Error {
                    image: value,
                    reason: e.into(),
                });
            }
        };
        Ok(match value {
            DynamicImage::ImageLuma8(x) => {
                Image::<u8, 1>::new_vec(extract_vec(x, width_times_height)?, width, height).into()
            }
            DynamicImage::ImageLuma16(x) => {
                Image::<u16, 1>::new_vec(extract_vec(x, width_times_height)?, width, height).into()
            }
            DynamicImage::ImageLumaA8(x) => Image::<[u8; 2], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            DynamicImage::ImageLumaA16(x) => Image::<[u16; 2], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            DynamicImage::ImageRgb8(x) => Image::<[u8; 3], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            DynamicImage::ImageRgb16(x) => Image::<[u16; 3], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            DynamicImage::ImageRgb32F(x) => Image::<[f32; 3], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            DynamicImage::ImageRgba8(x) => Image::<[u8; 4], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            DynamicImage::ImageRgba16(x) => Image::<[u16; 4], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            DynamicImage::ImageRgba32F(x) => Image::<[f32; 4], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            _ => {
                return Err(IntoDynamicImage0_25Error {
                    image: value,
                    reason: IntoDynamicImage0_25ErrorReason::NonExhaustive,
                });
            }
        })
    }
}

fn extract_vec<TPixel: image_0_25::Pixel>(
    image: image_0_25::ImageBuffer<TPixel, Vec<TPixel::Subpixel>>,
    width_times_height: usize,
) -> Result<Vec<TPixel::Subpixel>, IntoDynamicImage0_25Error>
where
    DynamicImage: From<image_0_25::ImageBuffer<TPixel, Vec<TPixel::Subpixel>>>,
{
    let actual = image.len();
    let expected = width_times_height * TPixel::CHANNEL_COUNT as usize;
    if actual != expected {
        return Err(IntoDynamicImage0_25Error {
            image: image.into(),
            reason: IncompatibleBufferSize {
                expected: width_times_height,
                actual,
            }
            .into(),
        });
    }
    let vec = image.into_vec();
    Ok(vec)
}

fn image_from_raw<P, T>(width: u32, height: u32, buffer: Vec<T>) -> ImageBuffer<P, Vec<T>>
where
    P: image_0_25::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("Incompatible buffer size")
}

fn ref_image<P, T>(width: u32, height: u32, buffer: &[T]) -> ImageBuffer<P, &[T]>
where
    P: image_0_25::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("Incompatible buffer size")
}

#[cfg(test)]
mod tests {

    use std::{io::Cursor, num::NonZeroU32};

    #[allow(unused_imports)] // Bug, probably because of crate renaming
    use image_0_25::DynamicImage;

    use crate::{DynamicRefImage0_25, Image};

    #[test]
    fn test_try_from_dynamic_luma_image() {
        let image = DynamicImage::new_luma8(100, 100);
        let dynamic_image = crate::DynamicImage::try_from(image).unwrap();
        assert_eq!(dynamic_image[0].width().get(), 100);
        assert_eq!(dynamic_image[0].height().get(), 100);
    }

    #[test]
    fn test_try_from_dynamic_rgb_image() {
        let image = DynamicImage::new_rgb16(100, 100);
        let dynamic_image = crate::DynamicImage::try_from(image).unwrap();
        assert_eq!(dynamic_image[0].width().get(), 100);
        assert_eq!(dynamic_image[0].height().get(), 100);
    }

    #[test]
    fn create_image_from_zero_width_fails() {
        let image: image_0_25::ImageBuffer<image_0_25::Luma<u8>, Vec<u8>> =
            image_0_25::ImageBuffer::from_raw(0, 1, vec![]).unwrap();
        crate::DynamicImage::try_from(DynamicImage::from(image)).unwrap_err();
    }

    #[test]
    fn create_image_from_zero_height_fails() {
        let image: image_0_25::ImageBuffer<image_0_25::Luma<u8>, Vec<u8>> =
            image_0_25::ImageBuffer::from_raw(1, 0, vec![]).unwrap();

        crate::DynamicImage::try_from(DynamicImage::from(image)).unwrap_err();
    }

    #[test]
    fn create_image_from_wrong_vec_len() {
        let image: image_0_25::ImageBuffer<image_0_25::Luma<u8>, Vec<u8>> =
            image_0_25::ImageBuffer::from_raw(1, 1, vec![0, 1]).unwrap();

        crate::DynamicImage::try_from(DynamicImage::from(image)).unwrap_err();
    }

    #[test]
    fn luma8_into_dynamic_and_back() {
        let mut expected = Cursor::new(Vec::new());
        let format = image_0_25::ImageFormat::Png;
        image_0_25::ImageBuffer::<image_0_25::Luma<u8>, Vec<u8>>::from_raw(1, 1, vec![1])
            .unwrap()
            .write_to(&mut expected, format)
            .unwrap();
        let expected = expected.into_inner();
        let image = Image::<u8, 1>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);

        assert_eq!(expected, test_encode(DynamicRefImage0_25::from(&image)));
        let dynamic = crate::DynamicImage::from(image);
        assert_eq!(
            expected,
            test_encode(DynamicRefImage0_25::try_from(&dynamic).unwrap())
        );
    }

    #[test]
    fn rgb16_into_dynamic_and_back() {
        let mut expected = Cursor::new(Vec::new());
        let format = image_0_25::ImageFormat::Png;
        image_0_25::ImageBuffer::<image_0_25::Rgb<u16>, Vec<u16>>::from_raw(1, 1, vec![0, 1, 2])
            .unwrap()
            .write_to(&mut expected, format)
            .unwrap();
        let expected = expected.into_inner();
        let image =
            Image::<[u16; 3], 1>::new_vec(vec![[0, 1, 2]], NonZeroU32::MIN, NonZeroU32::MIN);

        assert_eq!(expected, test_encode(DynamicRefImage0_25::from(&image)));
        let dynamic = crate::DynamicImage::from(image);
        assert_eq!(
            expected,
            test_encode(DynamicRefImage0_25::try_from(&dynamic).unwrap())
        );
    }

    fn test_encode(image: DynamicRefImage0_25<'_>) -> Vec<u8> {
        let mut expected = Cursor::new(Vec::new());
        let format = image_0_25::ImageFormat::Png;
        image.write_to(&mut expected, format).unwrap();
        expected.into_inner()
    }
}
