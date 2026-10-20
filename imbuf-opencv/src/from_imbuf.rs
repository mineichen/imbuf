use std::{
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
};

use imbuf::{DynamicImage, DynamicImageChannel, Image, ImageChannel, PixelType, RuntimePixelType};
use opencv::{
    boxed_ref::{BoxedRef, BoxedRefMut},
    core::{Code, DataType, Mat},
};

/// Borrows an [`imbuf`] buffer as an OpenCV [`Mat`] view without copying.
///
/// The `Mat` points at the `imbuf` allocation and is invalidated by the borrow
/// of the source. No owned conversion is provided in this direction: a `Mat`
/// cannot take ownership of an `imbuf` allocation without copying, so consumers
/// who need an owned `Mat` should copy the view explicitly.
pub trait ImbufAsOpencv {
    /// # Errors
    /// Fails if the buffer layout cannot be represented as a single `Mat`
    /// (for example a planar [`DynamicImage`] with more than one channel).
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>>;
}
pub trait ImbufAsOpencvMut: ImbufAsOpencv {
    /// Mutably borrows the buffer as an OpenCV [`Mat`] view without copying.
    ///
    /// Writes through the returned `Mat` land in the `imbuf` buffer.
    ///
    /// # Errors
    /// Fails if the buffer layout cannot be represented as a single `Mat`
    /// (for example a planar [`DynamicImage`] with more than one channel).
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>>;
}

// ---------------------------------------------------------------------------
// DynamicImage
// ---------------------------------------------------------------------------

fn ensure_single_channel(image: &DynamicImage) -> opencv::Result<()> {
    (image.len() == 1).then_some(()).ok_or_else(|| {
        opencv::Error::new(
            Code::StsUnsupportedFormat.into(),
            "unsupported format: a Mat can only represent a single channel",
        )
    })
}

impl ImbufAsOpencv for DynamicImage {
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        ensure_single_channel(self)?;
        match self.first() {
            DynamicImageChannel::U8(channel) => channel.as_opencv(),
            DynamicImageChannel::U16(channel) => channel.as_opencv(),
            DynamicImageChannel::F32(channel) => channel.as_opencv(),
        }
    }
}
impl ImbufAsOpencvMut for DynamicImage {
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        ensure_single_channel(self)?;
        match &mut self[0] {
            DynamicImageChannel::U8(channel) => channel.as_opencv_mut(),
            DynamicImageChannel::U16(channel) => channel.as_opencv_mut(),
            DynamicImageChannel::F32(channel) => channel.as_opencv_mut(),
        }
    }
}

// ---------------------------------------------------------------------------
// ImageChannel
// ---------------------------------------------------------------------------

impl<TP: RuntimePixelType> ImbufAsOpencv for ImageChannel<TP>
where
    TP::Primitive: DataType,
{
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.buffer_flat();
        let len = flat.len();
        let ptr = flat.as_ptr().cast_mut().cast::<c_void>();
        let mat = unsafe { create_unrestricted_mat::<TP>(dims, len, self.pixel_elements(), ptr) }?;
        Ok(BoxedRef::from(mat))
    }
}
impl<TP: RuntimePixelType> ImbufAsOpencvMut for ImageChannel<TP>
where
    TP::Primitive: DataType,
{
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.primitive_make_mut();
        let len = flat.len();
        let ptr = flat.as_mut_ptr().cast::<c_void>();
        let mat = unsafe { create_unrestricted_mat::<TP>(dims, len, self.pixel_elements(), ptr) }?;
        Ok(BoxedRefMut::from(mat))
    }
}

// ---------------------------------------------------------------------------
// Image<TP, 1>
// ---------------------------------------------------------------------------

impl<TP: PixelType> ImbufAsOpencv for Image<TP, 1>
where
    TP::Primitive: DataType,
{
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.buffer_flat();
        let ptr = flat.as_ptr().cast_mut().cast::<c_void>();
        let mat = unsafe { create_unrestricted_mat::<TP>(dims, flat.len(), TP::ELEMENTS, ptr) }?;
        Ok(BoxedRef::from(mat))
    }
}
impl<TP: PixelType> ImbufAsOpencvMut for Image<TP, 1>
where
    TP::Primitive: DataType,
{
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.buffer_flat_mut();
        let ptr = flat.as_mut_ptr().cast::<c_void>();
        let mat = unsafe { create_unrestricted_mat::<TP>(dims, flat.len(), TP::ELEMENTS, ptr) }?;
        Ok(BoxedRefMut::from(mat))
    }
}

/// Rejects a buffer whose length is not exactly `height * width * elements`.
///
/// # Safety
/// Assumes that the returned `Mat`'s lifetime is restricted.
unsafe fn create_unrestricted_mat<TP: RuntimePixelType>(
    (width, height): (NonZeroU32, NonZeroU32),
    len: usize,
    pixel_elements: NonZeroU8,
    ptr: *mut c_void,
) -> opencv::Result<Mat>
where
    TP::Primitive: DataType,
{
    let expected =
        u64::from(height.get()) * u64::from(width.get()) * u64::from(pixel_elements.get());

    assert!(
        u64::try_from(len) == Ok(expected),
        "the buffer holds {len} values, but {expected} are needed"
    );

    let (rows, cols) = (height.get().cast_signed(), width.get().cast_signed());
    unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            rows,
            cols,
            opencv::core::CV_MAKETYPE(
                TP::Primitive::opencv_depth(),
                i32::from(pixel_elements.get()),
            ),
            ptr,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use imbuf::{DynamicImageChannel, PixelTypePrimitive};
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

        let borrowed = img.as_opencv().unwrap();
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
    fn as_opencv_mut_shares_buffer() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut img: Image<u8, 1> = Image::new_vec(vec![1, 2], width, height);
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let mut mat = img.as_opencv_mut().unwrap();
            mat.data_typed_mut::<u8>().unwrap()[0] = 42;
        }

        assert_eq!(img.buffer(), &[42, 2]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn as_opencv_shares_buffer_without_copying() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let img: Image<u8, 1> = Image::new_vec(vec![1, 2], width, height);
        let data_ptr = img.buffer().as_ptr() as usize;

        let owned_view = img.as_opencv().unwrap();
        let owned_mat = &owned_view;
        assert_eq!(owned_mat.rows(), 1);
        assert_eq!(owned_mat.cols(), 2);
        assert_eq!(
            owned_mat.data_typed::<u8>().unwrap().as_ptr() as usize,
            data_ptr
        );

        let img_ref: &Image<u8, 1> = &img;
        let borrowed_view = img_ref.as_opencv().unwrap();
        assert_eq!(
            borrowed_view.data_typed::<u8>().unwrap().as_ptr() as usize,
            data_ptr
        );
    }

    #[test]
    fn as_opencv_works_through_mut_receiver() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let mut img: Image<u8, 1> = Image::new_vec(vec![1, 2], width, height);
        let data_ptr = img.buffer().as_ptr() as usize;

        let img_ref: &mut Image<u8, 1> = &mut img;
        let view = img_ref.as_opencv().unwrap();
        assert_eq!(view.data_typed::<u8>().unwrap().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn interleaved_rgb_shares_buffer_as_cv_8uc3() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let img: Image<[u8; 3], 1> = Image::new_vec(vec![[1u8, 2, 3], [4, 5, 6]], width, height);
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_opencv().unwrap();
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
            let mut mat = img.as_opencv_mut().unwrap();
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

        let borrowed = img.as_opencv().unwrap();
        let mat = &borrowed;
        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 6);
        assert_eq!(mat.typ(), opencv::core::CV_MAKETYPE(opencv::core::CV_8U, 6));
        assert_eq!(mat.data() as usize, data_ptr);

        {
            let mut mat = img.as_opencv_mut().unwrap();
            mat.data_typed_mut::<VecN<u8, 6>>().unwrap()[0] = VecN([9, 8, 7, 6, 5, 4]);
        }
        assert_eq!(img.buffer(), &[[9, 8, 7, 6, 5, 4], [7, 8, 9, 10, 11, 12]]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn dynamic_image_as_opencv_shares_buffer() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let image: Image<u8, 1> = Image::new_vec(vec![1, 2], width, height);
        let data_ptr = image.buffer().as_ptr() as usize;
        let dynamic = DynamicImage::from(image);

        let borrowed = dynamic.as_opencv().unwrap();
        assert_eq!(
            borrowed.data_typed::<u8>().unwrap().as_ptr() as usize,
            data_ptr
        );

        let DynamicImageChannel::U8(channel) = dynamic.first() else {
            panic!("Expected U8 channel");
        };
        assert_eq!(channel.buffer_flat(), &[1, 2]);
    }

    #[test]
    fn dynamic_image_as_opencv_mut_writes_through() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let image: Image<u8, 1> = Image::new_vec(vec![1, 2], width, height);
        let mut dynamic = DynamicImage::from(image);

        {
            let mut mat = dynamic.as_opencv_mut().unwrap();
            mat.data_typed_mut::<u8>().unwrap()[0] = 42;
        }

        let DynamicImageChannel::U8(channel) = dynamic.first() else {
            panic!("Expected U8 channel");
        };
        assert_eq!(channel.buffer_flat(), &[42, 2]);
    }

    #[test]
    fn dynamic_image_with_interleaved_rgb_shares_buffer_as_cv_8uc3() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(1).unwrap();
        let image: Image<[u8; 3], 1> = Image::new_vec(vec![[1u8, 2, 3], [4, 5, 6]], width, height);
        let data_ptr = image.buffer().as_ptr() as usize;
        let dynamic = DynamicImage::from(image);

        let borrowed = dynamic.as_opencv().unwrap();
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
}
