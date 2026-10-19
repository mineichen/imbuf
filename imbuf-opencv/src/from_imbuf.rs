use std::num::{NonZeroU8, NonZeroU32};

use imbuf::{DynamicImage, DynamicImageChannel, Image, ImageChannel, PixelType, RuntimePixelType};
use opencv::boxed_ref::{BoxedRef, BoxedRefMut};
use opencv::core::{DataType, Mat};

/// Borrows an [`imbuf`] buffer as an OpenCV [`Mat`] without copying.
///
/// The returned `Mat` borrows the `imbuf` data for as long as the reference is
/// held.
pub trait ImbufToOpencv {
    /// # Errors
    /// Fails if the buffer layout cannot be represented as a `Mat`.
    fn as_mat(&self) -> opencv::Result<BoxedRef<'_, Mat>>;

    /// # Errors
    /// Fails if the buffer layout cannot be represented as a `Mat`.
    fn as_mat_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>>;
}

impl ImbufToOpencv for DynamicImage {
    fn as_mat(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        ensure_layout(self)?;
        match self.first() {
            DynamicImageChannel::U8(c) => c.as_mat(),
            DynamicImageChannel::U16(c) => c.as_mat(),
            DynamicImageChannel::F32(c) => c.as_mat(),
        }
    }

    fn as_mat_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        ensure_layout(self)?;
        match &mut self[0] {
            DynamicImageChannel::U8(c) => c.as_mat_mut(),
            DynamicImageChannel::U16(c) => c.as_mat_mut(),
            DynamicImageChannel::F32(c) => c.as_mat_mut(),
        }
    }
}

fn ensure_layout(img: &DynamicImage) -> opencv::Result<()> {
    if img.len() == 1 && img.first().pixel_elements() == NonZeroU8::MIN {
        Ok(())
    } else {
        Err(opencv::Error::new(
            opencv::core::Code::StsUnsupportedFormat.into(),
            "unsupported format: expected a single U8, U16 or F32 channel",
        ))
    }
}

impl<TP> ImbufToOpencv for ImageChannel<TP>
where
    TP: RuntimePixelType,
    TP::Primitive: DataType,
{
    fn as_mat(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        let (width, height) = self.dimensions();
        Mat::new_rows_cols_with_data(
            height.get().cast_signed(),
            width.get().cast_signed(),
            self.buffer_flat(),
        )
    }

    fn as_mat_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        let (width, height) = self.dimensions();
        Mat::new_rows_cols_with_data_mut(
            height.get().cast_signed(),
            width.get().cast_signed(),
            self.primitive_make_mut(),
        )
    }
}

impl<TP> ImbufToOpencv for Image<TP, 1>
where
    TP: PixelType,
    TP::Primitive: DataType,
{
    fn as_mat(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.buffer_flat();
        // Safety: Boxes the Mat into BoxedRef to restrict lifetime
        // Casting to mut is ok, since BoxedRef doesn't allow mutable access

        let mat = unsafe {
            create_unrestricted_mat::<TP>(
                dims,
                flat.len(),
                flat.as_ptr().cast_mut().cast::<std::ffi::c_void>(),
            )
        }?;
        Ok(BoxedRef::from(mat))
    }

    fn as_mat_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.buffer_flat_mut();

        // Safety: Boxes the Mat into BoxedRefMut to restrict lifetime
        let mat = unsafe {
            create_unrestricted_mat::<TP>(
                dims,
                flat.len(),
                flat.as_mut_ptr().cast::<std::ffi::c_void>(),
            )
        }?;
        Ok(BoxedRefMut::from(mat))
    }
}

/// Rejects a buffer whose length is not exactly `height * width * elements`.
/// Safety: assumes, that the returned Mat's lifetime is restricted
unsafe fn create_unrestricted_mat<TP>(
    (width, height): (NonZeroU32, NonZeroU32),
    len: usize,
    ptr: *mut std::ffi::c_void,
) -> opencv::Result<Mat>
where
    TP: PixelType,
    TP::Primitive: DataType,
{
    let expected = u64::from(height.get())
        .strict_mul(u64::from(width.get()))
        .strict_mul(u64::from(TP::ELEMENTS.get()));

    assert!(
        u64::try_from(len) == Ok(expected),
        "the buffer holds {len} values, but {expected} are needed"
    );

    let (rows, cols) = (height.get().cast_signed(), width.get().cast_signed());
    unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            rows,
            cols,
            opencv::core::CV_MAKETYPE(TP::Primitive::opencv_depth(), i32::from(TP::ELEMENTS.get())),
            ptr,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use imbuf::PixelTypePrimitive;
    use opencv::core::{VecN, Vector};
    use opencv::prelude::{MatTraitConst, MatTraitConstManual, MatTraitManual};

    use super::*;

    fn imbuf_image_to_mat_preserves_dims_and_content<T>(
        pixels: Vec<T>,
        decode_png_to_vec: impl Fn(&[u8]) -> Vec<T>,
    ) where
        T: PixelTypePrimitive + opencv::core::DataType + std::fmt::Debug + PartialEq + 'static,
    {
        let width = NonZeroU32::new(3).unwrap();
        let height = NonZeroU32::new(2).unwrap();
        let img: Image<T, 1> = Image::new_vec(pixels, width, height);

        let borrowed = img.as_mat().unwrap();
        let mat = &borrowed;

        assert_eq!(img.width(), width);
        assert_eq!(img.height(), height);
        assert_eq!(mat.rows(), height.get() as i32);
        assert_eq!(mat.cols(), width.get() as i32);

        let mat_data = mat.data_typed::<T>().unwrap();
        assert_eq!(mat_data, img.buffer());
        assert_eq!(mat_data.as_ptr() as usize, img.buffer().as_ptr() as usize);

        // Encode the borrowed Mat as PNG, reload via the `image` crate, and compare decoded pixels
        // with the original image buffer.
        let mut png = Vector::<u8>::new();
        opencv::imgcodecs::imencode(".png", mat, &mut png, &Vector::new()).unwrap();
        let decoded = decode_png_to_vec(png.as_slice());
        assert_eq!(decoded, img.buffer());
    }

    #[test]
    fn imbuf_image_u8_to_mat_preserves_dims_and_content() {
        imbuf_image_to_mat_preserves_dims_and_content(vec![0u8, 1, 2, 3, 4, 5], |png| {
            match image::load_from_memory_with_format(png, image::ImageFormat::Png).unwrap() {
                image::DynamicImage::ImageLuma8(img) => img.into_vec(),
                other => panic!("Expected Luma8 PNG, got {other:?}"),
            }
        });
    }

    #[test]
    fn imbuf_image_u16_to_mat_preserves_dims_and_content() {
        imbuf_image_to_mat_preserves_dims_and_content(vec![10u16, 20, 30, 40, 50, 60], |png| {
            match image::load_from_memory_with_format(png, image::ImageFormat::Png).unwrap() {
                image::DynamicImage::ImageLuma16(img) => img.into_vec(),
                other => panic!("Expected Luma16 PNG, got {other:?}"),
            }
        });
    }

    #[test]
    fn as_mat_mut_shares_buffer() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut img: Image<u8, 1> = Image::new_vec(vec![1, 2], width, height);
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let mut mat = img.as_mat_mut().unwrap();
            mat.data_typed_mut::<u8>().unwrap()[0] = 42;
        }

        assert_eq!(img.buffer(), &[42, 2]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn interleaved_rgb_shares_buffer_as_cv_8uc3() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let img: Image<[u8; 3], 1> = Image::new_vec(vec![[1u8, 2, 3], [4, 5, 6]], width, height);
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_mat().unwrap();
        let mat = &borrowed;

        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 3);
        assert_eq!(mat.typ(), opencv::core::CV_8UC3);
        assert_eq!(mat.data() as usize, data_ptr);
        assert_eq!(
            mat.data_typed::<VecN<u8, 3>>().unwrap(),
            &[VecN([1, 2, 3]), VecN([4, 5, 6])]
        );
    }

    #[test]
    fn interleaved_rgb_mut_writes_through_to_imbuf() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut img: Image<[u8; 3], 1> =
            Image::new_vec(vec![[1u8, 2, 3], [4, 5, 6]], width, height);
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let mut mat = img.as_mat_mut().unwrap();
            mat.data_typed_mut::<VecN<u8, 3>>().unwrap()[0] = VecN([7, 8, 9]);
        }

        assert_eq!(img.buffer(), &[[7, 8, 9], [4, 5, 6]]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn interleaved_pixels_with_more_than_four_channels() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut img: Image<[u8; 6], 1> = Image::new_vec(
            vec![[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            width,
            height,
        );
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_mat().unwrap();
        let mat = &borrowed;
        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 6);
        assert_eq!(mat.typ(), opencv::core::CV_MAKETYPE(opencv::core::CV_8U, 6));
        assert_eq!(mat.data() as usize, data_ptr);

        {
            let mut mat = img.as_mat_mut().unwrap();
            mat.data_typed_mut::<VecN<u8, 6>>().unwrap()[0] = VecN([9, 8, 7, 6, 5, 4]);
        }
        assert_eq!(img.buffer(), &[[9, 8, 7, 6, 5, 4], [7, 8, 9, 10, 11, 12]]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
    }
}
