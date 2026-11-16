use image::{
    ImageError, ImageResult,
    error::{ImageFormatHint, UnsupportedError, UnsupportedErrorKind},
};

use crate::ImageIntoImbuf;

/// Decodes an image from `data` into an [`imbuf::DynamicImage`].
///
/// The format is guessed from the data. Only formats enabled on the `image` crate are supported.
/// To load TIFFs with their raw samples (e.g. `u32`, more than 4 samples per pixel or several
/// pages), use `imbuf_tiff::load_from_memory` instead.
///
/// # Errors
/// Returns the same [`ImageError`] as [`image::load_from_memory`]. Additionally fails with
/// [`ImageError::Unsupported`], if the decoded image has no `imbuf` representation (e.g. a zero
/// dimension).
pub fn load_from_memory(data: &[u8]) -> ImageResult<imbuf::DynamicImage> {
    image::load_from_memory(data)?
        .into_imbuf_dynamic()
        .map_err(|error| {
            let format = image::guess_format(data).map_or(ImageFormatHint::Unknown, Into::into);
            ImageError::Unsupported(UnsupportedError::from_format_and_kind(
                format,
                UnsupportedErrorKind::GenericFeature(error.to_string()),
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use testresult::TestResult;

    use super::*;

    #[test]
    fn load_png_luma8() -> TestResult {
        let mut png = Cursor::new(Vec::new());
        image::GrayImage::from_raw(2, 1, vec![1, 2])
            .ok_or("buffer too small")?
            .write_to(&mut png, image::ImageFormat::Png)?;

        let image = load_from_memory(png.get_ref())?;
        let image = imbuf::Image::<u8, 1>::try_from(image)?;
        assert_eq!(image.buffer(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn load_garbage_fails() {
        assert!(matches!(
            load_from_memory(&[1, 2, 3]),
            Err(ImageError::Unsupported(_))
        ));
    }
}
