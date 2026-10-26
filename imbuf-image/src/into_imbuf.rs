use std::num::NonZeroU32;

use image::{GenericImageView, ImageBuffer};
use imbuf::Image;

use crate::IncompatibleBufferSize;

#[derive(thiserror::Error, Debug)]
#[error("Cannot convert {image:?} into DynamicImage: {reason}")]
#[non_exhaustive]
pub struct IntoImbufError {
    pub image: image::DynamicImage,
    pub reason: IntoImbufErrorReason,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IntoImbufErrorReason {
    #[error("Neither height nor length can be 0")]
    ZeroDimension(#[from] std::num::TryFromIntError),
    #[error(transparent)]
    IncompatibleBufferSize(#[from] IncompatibleBufferSize),
    #[error(
        "::image::DynamicImage is non_exhaustive, so it could be extended in the future. Report this as a bug/extension request to imbuf"
    )]
    NonExhaustive,
}

/// Converts an [`image`] `DynamicImage` into an [`imbuf`] buffer.
pub trait ImageIntoImbuf {
    /// # Errors
    /// Fails if a dimension is zero or a buffer length does not match the
    /// dimensions.
    fn into_imbuf_dynamic(self) -> Result<imbuf::DynamicImage, IntoImbufError>;

    /// # Errors
    /// Fails for the same reasons as [`ImageIntoImbuf::into_imbuf_dynamic`], or if the
    /// layout does not match the requested [`imbuf::Image`] type.
    fn into_imbuf_specific<TP, const CHANNELS: usize>(
        self,
    ) -> Result<imbuf::Image<TP, CHANNELS>, String>
    where
        TP: imbuf::PixelType,
        imbuf::Image<TP, CHANNELS>: TryFrom<imbuf::DynamicImage, Error: std::fmt::Display>;
}

impl ImageIntoImbuf for image::DynamicImage {
    fn into_imbuf_dynamic(self) -> Result<imbuf::DynamicImage, IntoImbufError> {
        let (width, height) = self.dimensions();
        let width_times_height = width as usize * height as usize;
        let width = match NonZeroU32::try_from(width) {
            Ok(width) => width,
            Err(e) => {
                return Err(IntoImbufError {
                    image: self,
                    reason: e.into(),
                });
            }
        };

        let height = match NonZeroU32::try_from(height) {
            Ok(height) => height,
            Err(e) => {
                return Err(IntoImbufError {
                    image: self,
                    reason: e.into(),
                });
            }
        };
        Ok(match self {
            image::DynamicImage::ImageLuma8(x) => {
                Image::<u8, 1>::new_vec(extract_vec(x, width_times_height)?, width, height).into()
            }
            image::DynamicImage::ImageLuma16(x) => {
                Image::<u16, 1>::new_vec(extract_vec(x, width_times_height)?, width, height).into()
            }
            image::DynamicImage::ImageLumaA8(x) => Image::<[u8; 2], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            image::DynamicImage::ImageLumaA16(x) => Image::<[u16; 2], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            image::DynamicImage::ImageRgb8(x) => Image::<[u8; 3], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            image::DynamicImage::ImageRgb16(x) => Image::<[u16; 3], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            image::DynamicImage::ImageRgb32F(x) => Image::<[f32; 3], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            image::DynamicImage::ImageRgba8(x) => Image::<[u8; 4], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            image::DynamicImage::ImageRgba16(x) => Image::<[u16; 4], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            image::DynamicImage::ImageRgba32F(x) => Image::<[f32; 4], 1>::new_vec_flat(
                extract_vec(x, width_times_height)?,
                width,
                height,
            )
            .into(),
            _ => {
                return Err(IntoImbufError {
                    image: self,
                    reason: IntoImbufErrorReason::NonExhaustive,
                });
            }
        })
    }

    fn into_imbuf_specific<TP, const CHANNELS: usize>(
        self,
    ) -> Result<imbuf::Image<TP, CHANNELS>, String>
    where
        TP: imbuf::PixelType,
        imbuf::Image<TP, CHANNELS>: TryFrom<imbuf::DynamicImage, Error: std::fmt::Display>,
    {
        let dynamic = self.into_imbuf_dynamic().map_err(|e| e.to_string())?;
        imbuf::Image::try_from(dynamic).map_err(|e| e.to_string())
    }
}

fn extract_vec<TPixel: image::Pixel>(
    image: ImageBuffer<TPixel, Vec<TPixel::Subpixel>>,
    width_times_height: usize,
) -> Result<Vec<TPixel::Subpixel>, IntoImbufError>
where
    image::DynamicImage: From<ImageBuffer<TPixel, Vec<TPixel::Subpixel>>>,
{
    let actual = image.len();
    let expected = width_times_height * TPixel::CHANNEL_COUNT as usize;
    if actual != expected {
        return Err(IntoImbufError {
            image: image.into(),
            reason: IncompatibleBufferSize { expected, actual }.into(),
        });
    }
    let vec = image.into_vec();
    Ok(vec)
}

#[cfg(test)]
mod tests {
    use imbuf::Image;

    use super::*;

    #[test]
    fn into_imbuf_dynamic_moves_image_buffer() {
        let image = image::DynamicImage::new_luma8(3, 1);
        let ptr = match &image {
            image::DynamicImage::ImageLuma8(buffer) => buffer.as_raw().as_ptr(),
            _ => unreachable!(),
        };

        let dynamic_image = image.into_imbuf_dynamic().unwrap();
        let channel = dynamic_image
            .first()
            .try_cast::<u8>()
            .expect("expected one byte per pixel");

        assert_eq!(channel.buffer().as_ptr(), ptr);
    }

    #[test]
    fn into_imbuf_dynamic_luma() {
        let image = image::DynamicImage::new_luma8(100, 100);
        let dynamic_image = image.into_imbuf_dynamic().unwrap();
        assert_eq!(dynamic_image[0].width().get(), 100);
        assert_eq!(dynamic_image[0].height().get(), 100);
    }

    #[test]
    fn into_imbuf_dynamic_rgb() {
        let image = image::DynamicImage::new_rgb16(100, 100);
        let dynamic_image = image.into_imbuf_dynamic().unwrap();
        assert_eq!(dynamic_image[0].width().get(), 100);
        assert_eq!(dynamic_image[0].height().get(), 100);
    }

    #[test]
    fn into_imbuf_specific_luma() {
        let image = image::DynamicImage::new_luma8(2, 1);
        let typed: Image<u8, 1> = image.into_imbuf_specific().unwrap();
        assert_eq!(typed.buffer(), &[0, 0]);
    }

    #[test]
    fn create_image_from_zero_width_fails() {
        let image: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
            image::ImageBuffer::from_raw(0, 1, vec![]).unwrap();
        image::DynamicImage::from(image)
            .into_imbuf_dynamic()
            .unwrap_err();
    }

    #[test]
    fn create_image_from_zero_height_fails() {
        let image: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
            image::ImageBuffer::from_raw(1, 0, vec![]).unwrap();

        image::DynamicImage::from(image)
            .into_imbuf_dynamic()
            .unwrap_err();
    }

    #[test]
    fn create_image_from_wrong_vec_len() {
        let image: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
            image::ImageBuffer::from_raw(1, 1, vec![0, 1]).unwrap();

        image::DynamicImage::from(image)
            .into_imbuf_dynamic()
            .unwrap_err();
    }
}
