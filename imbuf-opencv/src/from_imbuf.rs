use std::{
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
};

use bytemuck::Pod;
use imbuf::{
    DynamicImageChannelView, DynamicImageView, ImageChannelView, ImageView,
    IncompatibleChannelError, PixelType, PixelTypePrimitive,
};
use opencv::{
    boxed_ref::{BoxedRef, BoxedRefMut},
    core::{Code, DataType, Mat},
};

/// Primitive sizes of the supported depths. A [`DynamicImageChannel`] only knows the size of its
/// primitive, so it is viewed as the primitive [`crate::OpencvIntoImbuf`] creates for that size.
const U8_SIZE: usize = size_of::<u8>();
const U16_SIZE: usize = size_of::<u16>();
const F32_SIZE: usize = size_of::<f32>();

/// Borrows an [`imbuf`] buffer as an `OpenCV` [`Mat`] view without copying.
///
/// The `Mat` points at the `imbuf` buffer and has no reference count. Like the `Mat` of
/// `Mat::new_rows_cols_with_data`, it's only returned inside a `BoxedRef`, which limits it to
/// the borrow of the source. Copy the view (e.g. with `Mat::try_clone`) to keep the pixels.
pub trait ImbufAsOpencv {
    /// # Errors
    /// Fails if the buffer layout cannot be represented as a single `Mat`
    /// (for example a planar [`DynamicImage`] with more than one channel), or if its width or
    /// height exceeds `i32::MAX`.
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>>;
}
pub trait ImbufAsOpencvMut: ImbufAsOpencv {
    /// Mutably borrows the buffer as an `OpenCV` [`Mat`] view without copying.
    ///
    /// Writes through the returned `Mat` land in the `imbuf` buffer, which is made exclusive
    /// first (copy on write). The view is limited to the borrow like [`ImbufAsOpencv::as_opencv`].
    /// `OpenCV` can write any value, so typed images require `Pod` primitives.
    ///
    /// # Errors
    /// Fails if the buffer layout cannot be represented as a single `Mat`
    /// (for example a planar [`DynamicImage`] with more than one channel), or if its width or
    /// height exceeds `i32::MAX`.
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>>;
}

fn ensure_single_channel(image: &DynamicImageView<'_>) -> opencv::Result<()> {
    (image.len() == 1).then_some(()).ok_or_else(|| {
        opencv::Error::new(
            Code::StsUnsupportedFormat.into(),
            "unsupported format: a Mat can only represent a single channel",
        )
    })
}

impl ImbufAsOpencv for DynamicImageView<'_> {
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        ensure_single_channel(self)?;
        self.first().as_opencv()
    }
}
impl ImbufAsOpencvMut for DynamicImageView<'_> {
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        ensure_single_channel(self)?;
        self[0].as_opencv_mut()
    }
}

fn unsupported_primitive(channel: &DynamicImageChannelView<'_>) -> opencv::Error {
    opencv::Error::new(
        Code::StsUnsupportedFormat.into(),
        format!(
            "unsupported format: primitives must have 1, 2 or 4 bytes and be aligned, got {}",
            channel.pixel_layout()
        ),
    )
}

fn incompatible_primitive(error: &IncompatibleChannelError) -> opencv::Error {
    opencv::Error::new(Code::StsUnsupportedFormat.into(), error.to_string())
}

impl ImbufAsOpencv for DynamicImageChannelView<'_> {
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        fn view<'b, P: PixelTypePrimitive + Pod + DataType>(
            channel: &'b DynamicImageChannelView<'_>,
        ) -> opencv::Result<BoxedRef<'b, Mat>> {
            let flat = channel
                .buffer_flat::<P>()
                .map_err(|e| incompatible_primitive(&e))?;
            let ptr = flat.as_ptr().cast_mut().cast::<c_void>();
            let (dims, pixel_elements) = (channel.dimensions(), channel.pixel_elements());
            // Safety: The BoxedRef limits the Mat to the borrow of the buffer
            let mat =
                unsafe { create_unrestricted_mat::<P>(dims, flat.len(), pixel_elements, ptr) }?;
            Ok(BoxedRef::from(mat))
        }

        match self.primitive_size().get() {
            U8_SIZE => view::<u8>(self),
            U16_SIZE => view::<u16>(self),
            F32_SIZE => view::<f32>(self),
            _ => Err(unsupported_primitive(self)),
        }
    }
}
impl ImbufAsOpencvMut for DynamicImageChannelView<'_> {
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        fn view<'b, P: PixelTypePrimitive + Pod + DataType>(
            channel: &'b mut DynamicImageChannelView<'_>,
        ) -> opencv::Result<BoxedRefMut<'b, Mat>> {
            let (dims, pixel_elements) = (channel.dimensions(), channel.pixel_elements());
            let flat = channel
                .make_mut_flat::<P>()
                .map_err(|e| incompatible_primitive(&e))?;
            let ptr = flat.as_mut_ptr().cast::<c_void>();
            // Safety: The BoxedRefMut limits the Mat to the mutable borrow of the buffer
            let mat =
                unsafe { create_unrestricted_mat::<P>(dims, flat.len(), pixel_elements, ptr) }?;
            Ok(BoxedRefMut::from(mat))
        }

        match self.primitive_size().get() {
            U8_SIZE => view::<u8>(self),
            U16_SIZE => view::<u16>(self),
            F32_SIZE => view::<f32>(self),
            _ => Err(unsupported_primitive(self)),
        }
    }
}

impl<TP: PixelType> ImbufAsOpencv for ImageChannelView<'_, TP>
where
    TP::Primitive: DataType,
{
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.buffer_flat();
        let len = flat.len();
        let ptr = flat.as_ptr().cast_mut().cast::<c_void>();
        let els = self.pixel_elements();
        // Safety: The BoxedRef limits the Mat to the borrow of the buffer
        let mat = unsafe { create_unrestricted_mat::<TP::Primitive>(dims, len, els, ptr) }?;
        Ok(BoxedRef::from(mat))
    }
}
impl<TP: PixelType> ImbufAsOpencvMut for ImageChannelView<'_, TP>
where
    TP::Primitive: DataType + Pod,
{
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        let dims = self.dimensions();
        let flat = self.make_mut_flat();
        let len = flat.len();
        let ptr = flat.as_mut_ptr().cast::<c_void>();
        let els = self.pixel_elements();

        // Safety: The BoxedRefMut limits the Mat to the mutable borrow of the buffer
        let mat = unsafe { create_unrestricted_mat::<TP::Primitive>(dims, len, els, ptr) }?;
        Ok(BoxedRefMut::from(mat))
    }
}

impl<TP: PixelType> ImbufAsOpencv for ImageView<'_, TP, 1>
where
    TP::Primitive: DataType,
{
    fn as_opencv(&self) -> opencv::Result<BoxedRef<'_, Mat>> {
        self.first().as_opencv()
    }
}
impl<TP: PixelType> ImbufAsOpencvMut for ImageView<'_, TP, 1>
where
    TP::Primitive: DataType + Pod,
{
    fn as_opencv_mut(&mut self) -> opencv::Result<BoxedRefMut<'_, Mat>> {
        self.first_mut().as_opencv_mut()
    }
}

/// The rows and columns of a `Mat` with the dimensions of an image. `OpenCV` uses `int` sizes, so
/// larger dimensions are rejected (they would wrap to negative sizes).
fn mat_size((width, height): (NonZeroU32, NonZeroU32)) -> opencv::Result<(i32, i32)> {
    match (i32::try_from(height.get()), i32::try_from(width.get())) {
        (Ok(rows), Ok(cols)) => Ok((rows, cols)),
        _ => Err(opencv::Error::new(
            Code::StsOutOfRange.into(),
            format!("A Mat can't have {width}x{height} pixels, as its dimensions are `int`s"),
        )),
    }
}

/// Rejects a buffer whose length is not exactly `height * width * elements`.
///
/// # Safety
/// `ptr` has to point to `len` values of `P`. The returned `Mat` has no reference count, so it
/// may only be returned inside a `BoxedRef`, which is limited to the borrow of the buffer, or a
/// `BoxedRefMut` for an exclusive borrow, if it's written.
unsafe fn create_unrestricted_mat<P: DataType>(
    (width, height): (NonZeroU32, NonZeroU32),
    len: usize,
    pixel_elements: NonZeroU8,
    ptr: *mut c_void,
) -> opencv::Result<Mat> {
    let expected =
        u64::from(height.get()) * u64::from(width.get()) * u64::from(pixel_elements.get());

    assert!(
        u64::try_from(len) == Ok(expected),
        "the buffer holds {len} values, but {expected} are needed"
    );

    let (rows, cols) = mat_size((width, height))?;
    unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            rows,
            cols,
            opencv::core::CV_MAKETYPE(P::opencv_depth(), i32::from(pixel_elements.get())),
            ptr,
        )
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::num::NonZeroU32;

    use imbuf::{DynamicImage, DynamicImageChannel, Image, PixelTypePrimitive};
    use opencv::core::{VecN, Vector};
    use opencv::prelude::{MatTraitConst, MatTraitConstManual, MatTraitManual};
    use testresult::TestResult;

    use super::*;

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    const NZ_3: NonZeroU32 = NonZeroU32::new(3).unwrap();

    fn imbuf_image_to_mat_preserves_dims_and_content<T>(
        pixels: Vec<T>,
        decode_png_to_vec: impl Fn(&[u8]) -> TestResult<Vec<T>>,
    ) -> TestResult
    where
        T: PixelTypePrimitive + opencv::core::DataType + std::fmt::Debug + PartialEq + 'static,
    {
        let img: Image<T, 1> = Image::new_vec(pixels, NZ_3, NZ_2)?;

        let borrowed = img.as_opencv()?;
        let mat = &borrowed;

        assert_eq!(img.width(), NZ_3);
        assert_eq!(img.height(), NZ_2);
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.rows(), 2);

        let mat_data = mat.data_typed::<T>()?;
        assert_eq!(mat_data, img.buffer());
        assert_eq!(mat_data.as_ptr() as usize, img.buffer().as_ptr() as usize);

        // Encode the borrowed Mat as PNG, reload via the `image` crate, and compare decoded pixels
        // with the original image buffer.
        let mut png = Vector::<u8>::new();
        opencv::imgcodecs::imencode(".png", mat, &mut png, &Vector::new())?;
        let decoded = decode_png_to_vec(png.as_slice())?;
        assert_eq!(decoded, img.buffer());
        Ok(())
    }

    #[test]
    fn imbuf_image_u8_to_mat_preserves_dims_and_content() -> TestResult {
        imbuf_image_to_mat_preserves_dims_and_content(vec![0u8, 1, 2, 3, 4, 5], |png| {
            match image::load_from_memory_with_format(png, image::ImageFormat::Png)? {
                image::DynamicImage::ImageLuma8(img) => Ok(img.into_vec()),
                other => Err(format!("Expected Luma8 PNG, got {other:?}").into()),
            }
        })
    }

    #[test]
    fn imbuf_image_u16_to_mat_preserves_dims_and_content() -> TestResult {
        imbuf_image_to_mat_preserves_dims_and_content(vec![10u16, 20, 30, 40, 50, 60], |png| {
            match image::load_from_memory_with_format(png, image::ImageFormat::Png)? {
                image::DynamicImage::ImageLuma16(img) => Ok(img.into_vec()),
                other => Err(format!("Expected Luma16 PNG, got {other:?}").into()),
            }
        })
    }

    #[test]
    fn as_opencv_mut_shares_buffer() -> TestResult {
        let mut img: Image<u8, 1> = Image::new_vec(vec![1, 2], NZ_2, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let mut mat = img.as_opencv_mut()?;
            mat.data_typed_mut::<u8>()?[0] = 42;
        }

        assert_eq!(img.buffer(), &[42, 2]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn as_opencv_shares_buffer_without_copying() -> TestResult {
        let img: Image<u8, 1> = Image::new_vec(vec![1, 2], NZ_2, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        let owned_view = img.as_opencv()?;
        let owned_mat = &owned_view;
        assert_eq!(owned_mat.rows(), 1);
        assert_eq!(owned_mat.cols(), 2);
        assert_eq!(owned_mat.data_typed::<u8>()?.as_ptr() as usize, data_ptr);

        let img_ref: &Image<u8, 1> = &img;
        let borrowed_view = img_ref.as_opencv()?;
        assert_eq!(
            borrowed_view.data_typed::<u8>()?.as_ptr() as usize,
            data_ptr
        );
        Ok(())
    }

    #[test]
    fn borrowed_image_as_opencv_mut_writes_into_slice() -> TestResult {
        let mut data = [1u8, 2];
        {
            let mut img = ImageView::<u8, 1>::new_slice_mut(&mut data, NZ_2, NZ_1)?;
            let mut mat = img.as_opencv_mut()?;
            mat.data_typed_mut::<u8>()?[1] = 42;
        }
        assert_eq!(data, [1, 42]);
        Ok(())
    }

    #[test]
    fn as_opencv_works_through_mut_receiver() -> TestResult {
        let mut img: Image<u8, 1> = Image::new_vec(vec![1, 2], NZ_2, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        let img_ref: &mut Image<u8, 1> = &mut img;
        let view = img_ref.as_opencv()?;
        assert_eq!(view.data_typed::<u8>()?.as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn interleaved_rgb_shares_buffer_as_cv_8uc3() -> TestResult {
        let img: Image<[u8; 3], 1> = Image::new_vec(vec![[1u8, 2, 3], [4, 5, 6]], NZ_2, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_opencv()?;
        let mat = &borrowed;

        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 3);
        assert_eq!(mat.typ(), opencv::core::CV_8UC3);
        assert_eq!(mat.data() as usize, data_ptr);
        assert_eq!(
            mat.data_typed::<VecN<u8, 3>>()?,
            &[VecN([1, 2, 3]), VecN([4, 5, 6])]
        );
        Ok(())
    }

    #[test]
    fn interleaved_rgb_mut_writes_through_to_imbuf() -> TestResult {
        let mut img: Image<[u8; 3], 1> = Image::new_vec(vec![[1u8, 2, 3], [4, 5, 6]], NZ_2, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        {
            let mut mat = img.as_opencv_mut()?;
            mat.data_typed_mut::<VecN<u8, 3>>()?[0] = VecN([7, 8, 9]);
        }

        assert_eq!(img.buffer(), &[[7, 8, 9], [4, 5, 6]]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn interleaved_pixels_with_more_than_four_channels() -> TestResult {
        let mut img: Image<[u8; 6], 1> =
            Image::new_vec(vec![[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], NZ_2, NZ_1)?;
        let data_ptr = img.buffer().as_ptr() as usize;

        let borrowed = img.as_opencv()?;
        let mat = &borrowed;
        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 6);
        assert_eq!(mat.typ(), opencv::core::CV_MAKETYPE(opencv::core::CV_8U, 6));
        assert_eq!(mat.data() as usize, data_ptr);

        {
            let mut mat = img.as_opencv_mut()?;
            mat.data_typed_mut::<VecN<u8, 6>>()?[0] = VecN([9, 8, 7, 6, 5, 4]);
        }
        assert_eq!(img.buffer(), &[[9, 8, 7, 6, 5, 4], [7, 8, 9, 10, 11, 12]]);
        assert_eq!(img.buffer().as_ptr() as usize, data_ptr);
        Ok(())
    }

    #[test]
    fn dynamic_image_as_opencv_shares_buffer() -> TestResult {
        let image: Image<u8, 1> = Image::new_vec(vec![1, 2], NZ_2, NZ_1)?;
        let data_ptr = image.buffer().as_ptr() as usize;
        let dynamic = DynamicImage::from(image);

        let borrowed = dynamic.as_opencv()?;
        assert_eq!(borrowed.data_typed::<u8>()?.as_ptr() as usize, data_ptr);

        let channel = dynamic.first().try_cast::<u8>()?;
        assert_eq!(channel.buffer_flat(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn dynamic_image_as_opencv_mut_writes_through() -> TestResult {
        let image: Image<u8, 1> = Image::new_vec(vec![1, 2], NZ_2, NZ_1)?;
        let mut dynamic = DynamicImage::from(image);

        {
            let mut mat = dynamic.as_opencv_mut()?;
            mat.data_typed_mut::<u8>()?[0] = 42;
        }

        let channel = dynamic.first().try_cast::<u8>()?;
        assert_eq!(channel.buffer_flat(), &[42, 2]);
        Ok(())
    }

    #[test]
    fn dynamic_image_with_interleaved_rgb_shares_buffer_as_cv_8uc3() -> TestResult {
        let image: Image<[u8; 3], 1> = Image::new_vec(vec![[1u8, 2, 3], [4, 5, 6]], NZ_2, NZ_1)?;
        let data_ptr = image.buffer().as_ptr() as usize;
        let dynamic = DynamicImage::from(image);

        let borrowed = dynamic.as_opencv()?;
        let mat = &borrowed;

        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 3);
        assert_eq!(mat.typ(), opencv::core::CV_8UC3);
        assert_eq!(mat.data() as usize, data_ptr);
        assert_eq!(
            mat.data_typed::<VecN<u8, 3>>()?,
            &[VecN([1, 2, 3]), VecN([4, 5, 6])]
        );
        Ok(())
    }

    #[test]
    fn dynamic_channel_with_unsupported_primitive_size_is_rejected() -> TestResult {
        #[derive(Clone, Copy, PartialEq)]
        #[repr(transparent)]
        struct F64(f64);
        unsafe impl bytemuck::Zeroable for F64 {}
        unsafe impl bytemuck::Pod for F64 {}
        impl PixelTypePrimitive for F64 {}

        let mut channel = DynamicImageChannel::new_vec(vec![F64(1.)], NZ_1, NZ_1)?;

        assert_matches!(
            channel.as_opencv(),
            Err(e) if e.code_as_enum() == Some(Code::StsUnsupportedFormat)
        );
        assert_matches!(
            channel.as_opencv_mut(),
            Err(e) if e.code_as_enum() == Some(Code::StsUnsupportedFormat)
        );
        Ok(())
    }

    #[test]
    fn dimensions_beyond_int_are_rejected() -> TestResult {
        let max = NonZeroU32::new(i32::MAX.unsigned_abs()).ok_or("zero")?;
        let too_large = max.checked_add(1).ok_or("overflow")?;
        assert_eq!(mat_size((max, NZ_1))?, (1, i32::MAX));
        assert_eq!(mat_size((NZ_1, max))?, (i32::MAX, 1));
        for dimensions in [
            (too_large, NZ_1),
            (NZ_1, too_large),
            (NonZeroU32::MAX, NonZeroU32::MAX),
        ] {
            assert_matches!(
                mat_size(dimensions),
                Err(e) if e.code_as_enum() == Some(Code::StsOutOfRange)
            );
        }
        Ok(())
    }
}
