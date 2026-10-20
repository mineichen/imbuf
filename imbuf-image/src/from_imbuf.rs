use std::num::NonZeroU8;

use image::ImageBuffer;
use imbuf::{DynamicImageChannel, Image, ImageChannel, RuntimePixelType};

use crate::{
    non_zero_u8,
    ref_image::{DynamicRefImage, DynamicRefImageMut},
};

#[derive(Debug, thiserror::Error)]
#[error(
    "Incompatible DynamicImage: channels={channels:?}(must always be 1), pixel_elements={pixel_elements:?}"
)]
pub struct IncompatibleLayoutError {
    channels: Option<NonZeroU8>,
    pixel_elements: Option<NonZeroU8>,
}

/// Borrows an [`imbuf`] buffer as an [`image`] value.
pub trait ImbufAsRustImage {
    /// Borrows the buffer as a shared [`DynamicRefImage`] without copying.
    ///
    /// # Errors
    /// Fails if the buffer layout cannot be represented as an `image` type.
    fn as_rust_image_dynamic(&self) -> Result<DynamicRefImage<'_>, IncompatibleLayoutError>;
}

/// Mutably borrows an [`imbuf`] buffer as an [`image`] value.
pub trait ImbufAsRustImageMut {
    /// Borrows the buffer as a [`DynamicRefImageMut`] without copying.
    ///
    /// # Errors
    /// Fails if the buffer layout cannot be represented as an `image` type.
    fn as_rust_image_dynamic_mut(
        &mut self,
    ) -> Result<DynamicRefImageMut<'_>, IncompatibleLayoutError>;
}

macro_rules! impl_from_borrowed_image_buffer {
    ($($variant:ident),+ $(,)?) => {
        impl ImbufAsRustImage for image::DynamicImage {
            fn as_rust_image_dynamic(
                &self,
            ) -> Result<DynamicRefImage<'_>, IncompatibleLayoutError> {
                match self {
                    $(image::DynamicImage::$variant(x) => Ok(x.into()),)+
                    _ => Err(IncompatibleLayoutError {
                        channels: None,
                        pixel_elements: None,
                    }),
                }
            }
        }

        impl ImbufAsRustImageMut for image::DynamicImage {
            fn as_rust_image_dynamic_mut(
                &mut self,
            ) -> Result<DynamicRefImageMut<'_>, IncompatibleLayoutError> {
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

impl<'a> TryFrom<&'a imbuf::DynamicImage> for DynamicRefImage<'a> {
    type Error = IncompatibleLayoutError;

    fn try_from(value: &'a imbuf::DynamicImage) -> Result<Self, Self::Error> {
        let channel = value.first();
        let pixel_elements = channel.pixel_elements();

        if value.len() != 1 {
            return Err(IncompatibleLayoutError {
                channels: Some(non_zero_u8(value.len_nonzero().get())),
                pixel_elements: Some(pixel_elements),
            });
        }

        match (channel, pixel_elements.get()) {
            (DynamicImageChannel::U8(x), 1) => Ok(DynamicRefImage::ImageLuma8(ref_image(x))),
            (DynamicImageChannel::U8(x), 2) => Ok(DynamicRefImage::ImageLumaA8(ref_image(x))),
            (DynamicImageChannel::U8(x), 3) => Ok(DynamicRefImage::ImageRgb8(ref_image(x))),
            (DynamicImageChannel::U8(x), 4) => Ok(DynamicRefImage::ImageRgba8(ref_image(x))),
            (DynamicImageChannel::U16(x), 1) => Ok(DynamicRefImage::ImageLuma16(ref_image(x))),
            (DynamicImageChannel::U16(x), 2) => Ok(DynamicRefImage::ImageLumaA16(ref_image(x))),
            (DynamicImageChannel::U16(x), 3) => Ok(DynamicRefImage::ImageRgb16(ref_image(x))),
            (DynamicImageChannel::U16(x), 4) => Ok(DynamicRefImage::ImageRgba16(ref_image(x))),
            (DynamicImageChannel::F32(x), 3) => Ok(DynamicRefImage::ImageRgb32F(ref_image(x))),
            (DynamicImageChannel::F32(x), 4) => Ok(DynamicRefImage::ImageRgba32F(ref_image(x))),
            (_, actual) => Err(IncompatibleLayoutError {
                channels: Some(non_zero_u8(value.len_nonzero().get())),
                pixel_elements: NonZeroU8::new(actual),
            }),
        }
    }
}

impl ImbufAsRustImage for imbuf::DynamicImage {
    fn as_rust_image_dynamic(&self) -> Result<DynamicRefImage<'_>, IncompatibleLayoutError> {
        DynamicRefImage::try_from(self)
    }
}

impl ImbufAsRustImageMut for imbuf::DynamicImage {
    fn as_rust_image_dynamic_mut(
        &mut self,
    ) -> Result<DynamicRefImageMut<'_>, IncompatibleLayoutError> {
        DynamicRefImageMut::try_from(self)
    }
}

impl<'a> TryFrom<&'a mut imbuf::DynamicImage> for DynamicRefImageMut<'a> {
    type Error = IncompatibleLayoutError;

    fn try_from(value: &'a mut imbuf::DynamicImage) -> Result<Self, Self::Error> {
        if value.len() != 1 {
            return Err(IncompatibleLayoutError {
                channels: Some(non_zero_u8(value.len())),
                pixel_elements: Some(value.first().pixel_elements()),
            });
        }
        let channels = non_zero_u8(value.len());
        let pixel_elements = value.first().pixel_elements().get();
        let channel = &mut value[0];
        type DynRefImgMut<'a> = DynamicRefImageMut<'a>;
        match (channel, pixel_elements) {
            (DynamicImageChannel::U8(x), 1) => Ok(DynRefImgMut::ImageLuma8(ref_image_mut(x))),
            (DynamicImageChannel::U8(x), 2) => Ok(DynRefImgMut::ImageLumaA8(ref_image_mut(x))),
            (DynamicImageChannel::U8(x), 3) => Ok(DynRefImgMut::ImageRgb8(ref_image_mut(x))),
            (DynamicImageChannel::U8(x), 4) => Ok(DynRefImgMut::ImageRgba8(ref_image_mut(x))),
            (DynamicImageChannel::U16(x), 1) => Ok(DynRefImgMut::ImageLuma16(ref_image_mut(x))),
            (DynamicImageChannel::U16(x), 2) => Ok(DynRefImgMut::ImageLumaA16(ref_image_mut(x))),
            (DynamicImageChannel::U16(x), 3) => Ok(DynRefImgMut::ImageRgb16(ref_image_mut(x))),
            (DynamicImageChannel::U16(x), 4) => Ok(DynRefImgMut::ImageRgba16(ref_image_mut(x))),
            (DynamicImageChannel::F32(x), 3) => Ok(DynRefImgMut::ImageRgb32F(ref_image_mut(x))),
            (DynamicImageChannel::F32(x), 4) => Ok(DynRefImgMut::ImageRgba32F(ref_image_mut(x))),
            (_, actual) => Err(IncompatibleLayoutError {
                channels: Some(channels),
                pixel_elements: Some(NonZeroU8::new(actual).unwrap()),
            }),
        }
    }
}

/// Converts an owned [`imbuf`] image with a statically known layout into an
/// owned [`image::DynamicImage`].
///
/// Uniquely owned buffers are moved without copying; shared buffers are copied.
/// The conversion is infallible because the typed source already matches an
/// `image` variant.
pub trait ImbufIntoRustImage {
    fn into_rust_image_dynamic(self) -> image::DynamicImage;
}

/// Fallibly converts an owned [`imbuf::DynamicImage`] into an owned
/// [`image::DynamicImage`].
///
/// The conversion fails if the dynamic image has more than one channel or a
/// depth/pixel layout that `image` cannot represent.
pub trait ImbufTryIntoRustImage {
    fn into_rust_image_dynamic(self) -> Result<image::DynamicImage, IncompatibleLayoutError>;
}

macro_rules! impl_imbuf_to_image {
    ($src:ty, $variant:ident) => {
        impl ImbufAsRustImage for Image<$src, 1> {
            fn as_rust_image_dynamic(
                &self,
            ) -> Result<DynamicRefImage<'_>, IncompatibleLayoutError> {
                Ok(DynamicRefImage::$variant(ref_image(self.first())))
            }
        }

        impl ImbufAsRustImageMut for Image<$src, 1> {
            fn as_rust_image_dynamic_mut(
                &mut self,
            ) -> Result<DynamicRefImageMut<'_>, IncompatibleLayoutError> {
                Ok(DynamicRefImageMut::$variant(ref_image_mut(
                    self.first_mut(),
                )))
            }
        }

        impl ImbufIntoRustImage for Image<$src, 1> {
            fn into_rust_image_dynamic(self) -> image::DynamicImage {
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

impl ImbufTryIntoRustImage for imbuf::DynamicImage {
    fn into_rust_image_dynamic(self) -> Result<image::DynamicImage, IncompatibleLayoutError> {
        enum Depth {
            U8,
            U16,
            F32,
        }
        let channels = non_zero_u8(self.len());
        let pixel_elements = self.first().pixel_elements();
        if self.len() != 1 {
            return Err(IncompatibleLayoutError {
                channels: Some(channels),
                pixel_elements: Some(pixel_elements),
            });
        }

        let depth = match self.first() {
            DynamicImageChannel::U8(_) => Depth::U8,
            DynamicImageChannel::U16(_) => Depth::U16,
            DynamicImageChannel::F32(_) => Depth::F32,
        };

        fn convert<TP, const N: usize>(
            dynamic: imbuf::DynamicImage,
            channels: NonZeroU8,
            pixel_elements: NonZeroU8,
        ) -> Result<image::DynamicImage, IncompatibleLayoutError>
        where
            TP: imbuf::PixelType,
            imbuf::Image<TP, N>: TryFrom<imbuf::DynamicImage> + ImbufIntoRustImage,
        {
            let image =
                imbuf::Image::<TP, N>::try_from(dynamic).map_err(|_| IncompatibleLayoutError {
                    channels: Some(channels),
                    pixel_elements: Some(pixel_elements),
                })?;
            Ok(image.into_rust_image_dynamic())
        }

        match (depth, pixel_elements.get()) {
            (Depth::U8, 1) => convert::<u8, 1>(self, channels, pixel_elements),
            (Depth::U8, 2) => convert::<[u8; 2], 1>(self, channels, pixel_elements),
            (Depth::U8, 3) => convert::<[u8; 3], 1>(self, channels, pixel_elements),
            (Depth::U8, 4) => convert::<[u8; 4], 1>(self, channels, pixel_elements),
            (Depth::U16, 1) => convert::<u16, 1>(self, channels, pixel_elements),
            (Depth::U16, 2) => convert::<[u16; 2], 1>(self, channels, pixel_elements),
            (Depth::U16, 3) => convert::<[u16; 3], 1>(self, channels, pixel_elements),
            (Depth::U16, 4) => convert::<[u16; 4], 1>(self, channels, pixel_elements),
            (Depth::F32, 3) => convert::<[f32; 3], 1>(self, channels, pixel_elements),
            (Depth::F32, 4) => convert::<[f32; 4], 1>(self, channels, pixel_elements),
            _ => Err(IncompatibleLayoutError {
                channels: Some(channels),
                pixel_elements: Some(pixel_elements),
            }),
        }
    }
}
/// The depth of an [`imbuf`] channel, matching the owned conversion arms.

fn image_from_raw<P, T>(width: u32, height: u32, buffer: Vec<T>) -> ImageBuffer<P, Vec<T>>
where
    P: image::Pixel<Subpixel = T>,
{
    ImageBuffer::from_raw(width, height, buffer).expect("There is no conversion which can fail")
}

fn ref_image<P, TP>(channel: &ImageChannel<TP>) -> ImageBuffer<P, &[TP::Primitive]>
where
    P: image::Pixel<Subpixel = TP::Primitive>,
    TP: RuntimePixelType,
{
    let (width, height) = channel.dimensions();
    ImageBuffer::from_raw(width.get(), height.get(), channel.buffer_flat())
        .expect("Incompatible buffer size")
}

fn ref_image_mut<P, TP>(channel: &mut ImageChannel<TP>) -> ImageBuffer<P, &mut [TP::Primitive]>
where
    P: image::Pixel<Subpixel = TP::Primitive>,
    TP: RuntimePixelType,
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

        assert_eq!(
            expected,
            test_encode(image.as_rust_image_dynamic().unwrap())
        );
        let dynamic = imbuf::DynamicImage::from(image);
        assert_eq!(
            expected,
            test_encode(dynamic.as_rust_image_dynamic().unwrap())
        );
    }

    #[test]
    fn into_image_round_trips() {
        let image = Image::<u8, 1>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        assert_eq!(
            test_encode(image.as_rust_image_dynamic().unwrap()),
            test_encode_owned(image.into_rust_image_dynamic())
        );
    }

    #[test]
    fn into_rust_image_shares_buffer() {
        let img =
            Image::<u8, 1>::new_vec(vec![1, 2, 3], NonZeroU32::new(3).unwrap(), NonZeroU32::MIN);
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_rust_image_dynamic().unwrap();
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
            let borrowed = img.as_rust_image_dynamic_mut().unwrap();
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

        let borrowed = img.as_rust_image_dynamic().unwrap();
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
        let borrowed = img_ref.as_rust_image_dynamic().unwrap();
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
            let borrowed = img.as_rust_image_dynamic_mut().unwrap();
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
