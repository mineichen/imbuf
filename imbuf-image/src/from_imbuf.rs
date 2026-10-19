use std::num::NonZeroU8;

use image::{ImageBuffer, Luma, LumaA, Rgb, Rgba};
use imbuf::{DynamicImageChannel, Image};

use crate::non_zero_u8;

/// A borrowed view of an [`imbuf`] image as an [`image`] `DynamicImage`-shaped
/// value.
///
/// Unlike [`image::DynamicImage`], this type can borrow the buffer, so
/// converting from an `imbuf` image never copies.
#[derive(Debug)]
pub enum DynamicRefImage<'a> {
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

/// A mutable borrowed view of an [`imbuf`] image as an [`image`]
/// `DynamicImage`-shaped value.
#[derive(Debug)]
pub enum DynamicRefImageMut<'a> {
    ImageLuma8(ImageBuffer<Luma<u8>, &'a mut [u8]>),
    ImageLuma16(ImageBuffer<Luma<u16>, &'a mut [u16]>),
    ImageLumaA8(ImageBuffer<LumaA<u8>, &'a mut [u8]>),
    ImageLumaA16(ImageBuffer<LumaA<u16>, &'a mut [u16]>),
    ImageRgb8(ImageBuffer<Rgb<u8>, &'a mut [u8]>),
    ImageRgb16(ImageBuffer<Rgb<u16>, &'a mut [u16]>),
    ImageRgb32F(ImageBuffer<Rgb<f32>, &'a mut [f32]>),
    ImageRgba8(ImageBuffer<Rgba<u8>, &'a mut [u8]>),
    ImageRgba16(ImageBuffer<Rgba<u16>, &'a mut [u16]>),
    ImageRgba32F(ImageBuffer<Rgba<f32>, &'a mut [f32]>),
}

impl DynamicRefImage<'_> {
    /// # Errors
    /// Forwards errors of `image::write_to`
    pub fn write_to<W: std::io::Write + std::io::Seek>(
        &self,
        mut buffer: W,
        format: image::ImageFormat,
    ) -> Result<(), image::ImageError> {
        match self {
            DynamicRefImage::ImageLuma8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageLuma16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageLumaA8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageLumaA16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgb8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgb16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgb32F(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgba8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgba16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgba32F(x) => x.write_to(&mut buffer, format),
        }
    }
}

impl DynamicRefImageMut<'_> {
    /// # Errors
    /// Forwards errors of `image::write_to`
    pub fn write_to<W: std::io::Write + std::io::Seek>(
        &self,
        mut buffer: W,
        format: image::ImageFormat,
    ) -> Result<(), image::ImageError> {
        match self {
            DynamicRefImageMut::ImageLuma8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageLuma16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageLumaA8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageLumaA16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgb8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgb16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgb32F(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgba8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgba16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgba32F(x) => x.write_to(&mut buffer, format),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IntoDynamicRefImageError {
    #[error(
        "Incompatible DynamicImage: channels={channels}(must always be 1), pixel_elements={pixel_elements}"
    )]
    IncompatibleLayout {
        channels: NonZeroU8,
        pixel_elements: NonZeroU8,
    },
}

/// Borrows an [`imbuf`] buffer as an [`image`] value without copying.
pub trait ImbufToImage {
    /// # Errors
    /// Fails if the buffer layout cannot be represented as an `image` type.
    fn as_image(&self) -> Result<DynamicRefImage<'_>, IntoDynamicRefImageError>;

    /// # Errors
    /// Fails if the buffer layout cannot be represented as an `image` type.
    fn as_image_mut(&mut self) -> Result<DynamicRefImageMut<'_>, IntoDynamicRefImageError>;
}

impl<'a> TryFrom<&'a imbuf::DynamicImage> for DynamicRefImage<'a> {
    type Error = IntoDynamicRefImageError;

    fn try_from(value: &'a imbuf::DynamicImage) -> Result<Self, Self::Error> {
        let channel = value.first();
        let pixel_elements = channel.pixel_elements();

        if value.len() != 1 {
            return Err(IntoDynamicRefImageError::IncompatibleLayout {
                channels: non_zero_u8(value.len()),
                pixel_elements,
            });
        }

        let width = channel.width().get();
        let height = channel.height().get();

        match (channel, pixel_elements.get()) {
            (DynamicImageChannel::U8(x), 1) => {
                Ok(DynamicRefImage::ImageLuma8(ref_image::<Luma<u8>, u8>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U8(x), 2) => {
                Ok(DynamicRefImage::ImageLumaA8(ref_image::<LumaA<u8>, u8>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U8(x), 3) => {
                Ok(DynamicRefImage::ImageRgb8(ref_image::<Rgb<u8>, u8>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U8(x), 4) => {
                Ok(DynamicRefImage::ImageRgba8(ref_image::<Rgba<u8>, u8>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U16(x), 1) => {
                Ok(DynamicRefImage::ImageLuma16(ref_image::<Luma<u16>, u16>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U16(x), 2) => {
                Ok(DynamicRefImage::ImageLumaA16(ref_image::<LumaA<u16>, u16>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U16(x), 3) => {
                Ok(DynamicRefImage::ImageRgb16(ref_image::<Rgb<u16>, u16>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::U16(x), 4) => {
                Ok(DynamicRefImage::ImageRgba16(ref_image::<Rgba<u16>, u16>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::F32(x), 3) => {
                Ok(DynamicRefImage::ImageRgb32F(ref_image::<Rgb<f32>, f32>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (DynamicImageChannel::F32(x), 4) => {
                Ok(DynamicRefImage::ImageRgba32F(ref_image::<Rgba<f32>, f32>(
                    width,
                    height,
                    x.buffer_flat(),
                )))
            }
            (_, actual) => Err(IntoDynamicRefImageError::IncompatibleLayout {
                channels: non_zero_u8(value.len()),
                pixel_elements: NonZeroU8::new(actual).unwrap(),
            }),
        }
    }
}

impl ImbufToImage for imbuf::DynamicImage {
    fn as_image(&self) -> Result<DynamicRefImage<'_>, IntoDynamicRefImageError> {
        DynamicRefImage::try_from(self)
    }

    fn as_image_mut(&mut self) -> Result<DynamicRefImageMut<'_>, IntoDynamicRefImageError> {
        if self.len() != 1 {
            return Err(IntoDynamicRefImageError::IncompatibleLayout {
                channels: non_zero_u8(self.len()),
                pixel_elements: self.first().pixel_elements(),
            });
        }
        let channels = non_zero_u8(self.len());
        let pixel_elements = self.first().pixel_elements().get();
        let channel = &mut self[0];
        let width = channel.width().get();
        let height = channel.height().get();
        match (channel, pixel_elements) {
            (DynamicImageChannel::U8(x), 1) => {
                Ok(DynamicRefImageMut::ImageLuma8(
                    ref_image_mut::<Luma<u8>, u8>(width, height, x.primitive_make_mut()),
                ))
            }
            (DynamicImageChannel::U8(x), 2) => {
                Ok(DynamicRefImageMut::ImageLumaA8(ref_image_mut::<
                    LumaA<u8>,
                    u8,
                >(
                    width,
                    height,
                    x.primitive_make_mut(),
                )))
            }
            (DynamicImageChannel::U8(x), 3) => {
                Ok(DynamicRefImageMut::ImageRgb8(ref_image_mut::<Rgb<u8>, u8>(
                    width,
                    height,
                    x.primitive_make_mut(),
                )))
            }
            (DynamicImageChannel::U8(x), 4) => {
                Ok(DynamicRefImageMut::ImageRgba8(
                    ref_image_mut::<Rgba<u8>, u8>(width, height, x.primitive_make_mut()),
                ))
            }
            (DynamicImageChannel::U16(x), 1) => {
                Ok(DynamicRefImageMut::ImageLuma16(ref_image_mut::<
                    Luma<u16>,
                    u16,
                >(
                    width,
                    height,
                    x.primitive_make_mut(),
                )))
            }
            (DynamicImageChannel::U16(x), 2) => {
                Ok(DynamicRefImageMut::ImageLumaA16(ref_image_mut::<
                    LumaA<u16>,
                    u16,
                >(
                    width,
                    height,
                    x.primitive_make_mut(),
                )))
            }
            (DynamicImageChannel::U16(x), 3) => Ok(DynamicRefImageMut::ImageRgb16(
                ref_image_mut::<Rgb<u16>, u16>(width, height, x.primitive_make_mut()),
            )),
            (DynamicImageChannel::U16(x), 4) => {
                Ok(DynamicRefImageMut::ImageRgba16(ref_image_mut::<
                    Rgba<u16>,
                    u16,
                >(
                    width,
                    height,
                    x.primitive_make_mut(),
                )))
            }
            (DynamicImageChannel::F32(x), 3) => {
                Ok(DynamicRefImageMut::ImageRgb32F(ref_image_mut::<
                    Rgb<f32>,
                    f32,
                >(
                    width,
                    height,
                    x.primitive_make_mut(),
                )))
            }
            (DynamicImageChannel::F32(x), 4) => {
                Ok(DynamicRefImageMut::ImageRgba32F(ref_image_mut::<
                    Rgba<f32>,
                    f32,
                >(
                    width,
                    height,
                    x.primitive_make_mut(),
                )))
            }
            (_, actual) => Err(IntoDynamicRefImageError::IncompatibleLayout {
                channels,
                pixel_elements: NonZeroU8::new(actual).unwrap(),
            }),
        }
    }
}

macro_rules! impl_imbuf_to_image {
    ($src:ty, $pixel:ty, $variant:ident) => {
        impl ImbufToImage for Image<$src, 1> {
            fn as_image(&self) -> Result<DynamicRefImage<'_>, IntoDynamicRefImageError> {
                let (width, height) = self.dimensions();
                Ok(DynamicRefImage::$variant(ref_image(
                    width.get(),
                    height.get(),
                    self.buffer_flat(),
                )))
            }

            fn as_image_mut(&mut self) -> Result<DynamicRefImageMut<'_>, IntoDynamicRefImageError> {
                let (width, height) = self.dimensions();
                Ok(DynamicRefImageMut::$variant(ref_image_mut(
                    width.get(),
                    height.get(),
                    self.buffer_flat_mut(),
                )))
            }
        }
    };
}

impl_imbuf_to_image!(u8, Luma<u8>, ImageLuma8);
impl_imbuf_to_image!(u16, Luma<u16>, ImageLuma16);
impl_imbuf_to_image!([u8; 2], LumaA<u8>, ImageLumaA8);
impl_imbuf_to_image!([u16; 2], LumaA<u16>, ImageLumaA16);
impl_imbuf_to_image!([u8; 3], Rgb<u8>, ImageRgb8);
impl_imbuf_to_image!([u16; 3], Rgb<u16>, ImageRgb16);
impl_imbuf_to_image!([f32; 3], Rgb<f32>, ImageRgb32F);
impl_imbuf_to_image!([u8; 4], Rgba<u8>, ImageRgba8);
impl_imbuf_to_image!([u16; 4], Rgba<u16>, ImageRgba16);
impl_imbuf_to_image!([f32; 4], Rgba<f32>, ImageRgba32F);

/// Conversions from an owned [`imbuf`] image into an owned [`image`]
/// `DynamicImage`.
///
/// This is a trait (rather than a `From` impl) because both the source and
/// target types are foreign to this crate; Rust's orphan rules require a local
/// trait for such conversions.
pub trait IntoImage {
    fn into_image(self) -> image::DynamicImage;
}

macro_rules! impl_into_image {
    ($src:ty, $pixel:ty, $variant:ident) => {
        impl IntoImage for Image<$src, 1> {
            fn into_image(self) -> image::DynamicImage {
                let (width, height) = self.dimensions();
                let [channel] = <[_; 1]>::from(self);
                let buffer = channel.into_vec_flat();
                image::DynamicImage::$variant(image_from_raw(width.get(), height.get(), buffer))
            }
        }
    };
}

impl_into_image!(u8, Luma<u8>, ImageLuma8);
impl_into_image!(u16, Luma<u16>, ImageLuma16);
impl_into_image!([u8; 2], LumaA<u8>, ImageLumaA8);
impl_into_image!([u16; 2], LumaA<u16>, ImageLumaA16);
impl_into_image!([u8; 3], Rgb<u8>, ImageRgb8);
impl_into_image!([u16; 3], Rgb<u16>, ImageRgb16);
impl_into_image!([f32; 3], Rgb<f32>, ImageRgb32F);
impl_into_image!([u8; 4], Rgba<u8>, ImageRgba8);
impl_into_image!([u16; 4], Rgba<u16>, ImageRgba16);
impl_into_image!([f32; 4], Rgba<f32>, ImageRgba32F);

fn image_from_raw<P, T>(width: u32, height: u32, buffer: Vec<T>) -> ImageBuffer<P, Vec<T>>
where
    P: image::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("Incompatible buffer size")
}

fn ref_image<P, T>(width: u32, height: u32, buffer: &[T]) -> ImageBuffer<P, &[T]>
where
    P: image::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("Incompatible buffer size")
}

fn ref_image_mut<P, T>(width: u32, height: u32, buffer: &mut [T]) -> ImageBuffer<P, &mut [T]>
where
    P: image::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("Incompatible buffer size")
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

        assert_eq!(expected, test_encode(image.as_image().unwrap()));
        let dynamic = imbuf::DynamicImage::from(image);
        assert_eq!(expected, test_encode(dynamic.as_image().unwrap()));
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

        assert_eq!(expected, test_encode(image.as_image().unwrap()));
        let dynamic = imbuf::DynamicImage::from(image);
        assert_eq!(expected, test_encode(dynamic.as_image().unwrap()));
    }

    #[test]
    fn into_image_round_trips() {
        let image = Image::<u8, 1>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        assert_eq!(
            test_encode(image.as_image().unwrap()),
            test_encode_owned(image.into_image())
        );
    }

    #[test]
    fn as_image_ref_shares_buffer() {
        let img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_image().unwrap();
        let DynamicRefImage::ImageLuma8(buffer) = borrowed else {
            panic!("Expected Luma8");
        };
        assert_eq!(buffer.as_raw(), &[1, 2, 3]);
        assert_eq!(buffer.as_raw().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn as_image_mut_shares_buffer() {
        let mut img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let borrowed = img.as_image_mut().unwrap();
            let DynamicRefImageMut::ImageLuma8(mut buffer) = borrowed else {
                panic!("Expected Luma8");
            };
            buffer.as_mut()[0] = 42;
        }

        assert_eq!(img.buffer(), &[42, 2, 3]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
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
