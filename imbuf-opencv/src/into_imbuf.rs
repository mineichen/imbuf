use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
};

use bytemuck::Pod;
use imbuf::{
    DynamicImage, DynamicImageChannel, DynamicImageChannelView, Image, ImageChannelVTable,
    IncompatibleImage, PixelType, PixelTypePrimitive,
};
use opencv::{
    core::{
        CV_8U, CV_16U, CV_32F, CV_MAT_DEPTH_MASK, Code, Mat, MatTraitConst, Range, UMatData,
        UMatDataTraitConst, Vector,
    },
    prelude::MatTraitConstManual,
    traits::{Boxed, OpenCVIntoExternContainer},
};

/// Converts an owned `OpenCV` [`Mat`] into an [`imbuf`] buffer without copying
/// pixel data (the `Mat` is moved into the imbuf allocation).
///
/// This is the only direction that can hand a buffer over: the resulting
/// [`DynamicImage`] owns the `Mat`'s memory through the `imbuf` channel vtable,
/// so no copy is needed.
pub trait OpencvIntoImbuf {
    /// # Errors
    /// Fails if the `Mat` is empty, not contiguous, not 2D/3D, or has an
    /// unsupported depth.
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImage>;

    /// # Errors
    /// Fails for the same reasons as [`OpencvIntoImbuf::into_imbuf_dynamic`], or
    /// if the `Mat` layout does not match the requested [`imbuf::Image`] type.
    fn into_imbuf_specific<TP: PixelType, const CHANNELS: usize>(
        self,
    ) -> opencv::Result<Image<TP, CHANNELS>>
    where
        Image<TP, CHANNELS>: TryFrom<DynamicImage, Error = IncompatibleImage<DynamicImage>>;
}

impl OpencvIntoImbuf for Mat {
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImage> {
        match self.typ() & CV_MAT_DEPTH_MASK {
            CV_8U => dynamic_image::<u8>(self),
            CV_16U => dynamic_image::<u16>(self),
            CV_32F => dynamic_image::<f32>(self),
            _ => Err(opencv::Error::new(
                Code::StsUnsupportedFormat.into(),
                "Mat depth must be CV_8U, CV_16U or CV_32F",
            )),
        }
    }

    fn into_imbuf_specific<TP: PixelType, const CHANNELS: usize>(
        self,
    ) -> opencv::Result<Image<TP, CHANNELS>>
    where
        Image<TP, CHANNELS>: TryFrom<DynamicImage, Error = IncompatibleImage<DynamicImage>>,
    {
        Image::try_from(self.into_imbuf_dynamic()?)
            .map_err(|e| opencv::Error::new(Code::StsUnsupportedFormat.into(), e.error.to_string()))
    }
}

fn dynamic_image<P: PixelTypePrimitive + Pod>(mat: Mat) -> opencv::Result<DynamicImage> {
    let dims = mat.dims();
    if !(2..=3).contains(&dims) || mat.data().is_null() {
        return Err(opencv::Error::new(
            Code::StsBadArg.into(),
            "Mat must be a non-empty 2d or 3d matrix",
        ));
    }
    if !mat.is_physically_contiguous()? {
        return Err(opencv::Error::new(
            Code::StsUnmatchedSizes.into(),
            "Mat is not continuous",
        ));
    }

    if dims == 2 {
        let width = nonzero_dimension(mat.cols())?;
        let height = nonzero_dimension(mat.rows())?;
        let pixel_elements = nonzero_pixel_elements(mat.channels())?;
        let channel = unsafe { into_channel::<P>(mat, width, height, pixel_elements) };
        Ok(DynamicImage::from_channels(channel, []))
    } else {
        if mat.channels() != 1 {
            return Err(opencv::Error::new(
                Code::StsUnsupportedFormat.into(),
                "A planar Mat must have exactly 1 channel",
            ));
        }
        let sizes = &*mat.mat_size();
        let planes = nonzero_pixel_elements(sizes[0])?;
        let width = nonzero_dimension(sizes[2])?;
        let height = nonzero_dimension(sizes[1])?;
        let channels = (0..planes.get())
            .map(|plane| planar_channel::<P>(&mat, i32::from(plane), width, height))
            .collect::<opencv::Result<Vec<_>>>()?;
        let mut channels = channels.into_iter();
        let Some(first) = channels.next() else {
            return Err(opencv::Error::new(
                Code::StsBadArg.into(),
                "Mat must have at least one plane",
            ));
        };
        Ok(DynamicImage::from_channels(first, channels))
    }
}

/// Mirrors the public, ABI-stable prefix of `cv::Mat`.
///
/// The generated `cv_Mat_propU` shim dereferences `u` and therefore crashes for
/// Mats over external data (`u == null`), so read the public field directly.
#[repr(C)]
struct MatHeader {
    flags: i32,
    dims: i32,
    rows: i32,
    cols: i32,
    data: *mut c_void,
    datastart: *const c_void,
    dataend: *const c_void,
    datalimit: *const c_void,
    allocator: *mut c_void,
    u: *mut c_void,
}

/// Returns `true` if this `Mat` header is the sole owner of its pixel buffer.
///
/// Mats over external data (`u == null`) have no refcount and are never
/// considered exclusive. `urefcount`/`mapcount` guard against a `UMat` or
/// device mapping aliasing the same allocation.
fn mat_buffer_is_exclusive(mat: &Mat) -> bool {
    let u_ptr = unsafe { (*mat.as_raw().cast::<MatHeader>()).u };
    if u_ptr.is_null() {
        return false;
    }
    // `UMatData` is generated as an owning boxed type; wrap it so dropping the
    // view does not delete the allocation owned by the `Mat`.
    let u = ManuallyDrop::new(unsafe { UMatData::from_raw(u_ptr) });
    u.refcount() == 1 && u.urefcount() == 0 && u.mapcount() == 0
}

/// Reborrows the `cv::Mat` owned by the channel without taking ownership.
unsafe fn mat_view(image: &DynamicImageChannelView<'_>) -> ManuallyDrop<Mat> {
    let data = DynamicImageChannelView::raw_data(image);
    ManuallyDrop::new(unsafe { Mat::from_raw(data.cast()) })
}

const fn vtable<P: PixelTypePrimitive + Pod>() -> &'static ImageChannelVTable {
    &ImageChannelVTable {
        clone: clone_mat_channel::<P>,
        make_mut: make_mat_channel_mut::<P>,
        drop: drop_mat_channel,
        bytes_per_primitive: bytes_per_primitive::<P>,
    }
}

/// # Safety
/// `mat` must be continuous and hold `width * height * pixel_elements` values of `P`
unsafe fn into_channel<P: PixelTypePrimitive + Pod>(
    mat: Mat,
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
) -> DynamicImageChannel {
    let ptr = mat.data();
    let data = mat.into_raw().cast::<()>();
    // Safety: The Mat keeps the buffer alive until the vtable drops it and the vtable is built
    // for `P`
    unsafe {
        DynamicImageChannel::from_raw_parts(ptr, width, height, pixel_elements, vtable::<P>(), data)
    }
}

fn planar_channel<P: PixelTypePrimitive + Pod>(
    mat: &Mat,
    plane: i32,
    width: NonZeroU32,
    height: NonZeroU32,
) -> opencv::Result<DynamicImageChannel> {
    let mut ranges = Vector::<Range>::new();
    ranges.push(Range::new(plane, plane + 1)?);
    ranges.push(Range::all()?);
    ranges.push(Range::all()?);
    let view = Mat::ranges(mat, &ranges)?.opencv_into_extern_container()?;
    Ok(unsafe { into_channel::<P>(view, width, height, NonZeroU8::MIN) })
}

extern "C-unwind" fn bytes_per_primitive<P>() -> NonZeroUsize {
    const { NonZeroUsize::new(size_of::<P>()).expect("Primitives are not zero sized") }
}

unsafe extern "C-unwind" fn clone_mat_channel<'a, P: PixelTypePrimitive + Pod>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    let mat = unsafe { mat_view(image) };
    match Mat::copy(&*mat).and_then(OpenCVIntoExternContainer::opencv_into_extern_container) {
        Ok(cloned) => unsafe {
            // Safety: The cloned Mat keeps the buffer alive until the vtable drops it and the
            // vtable is built for `P`
            DynamicImageChannel::from_raw_parts(
                cloned.data(),
                image.width(),
                image.height(),
                image.pixel_elements(),
                DynamicImageChannelView::raw_vtable(image),
                cloned.into_raw().cast::<()>(),
            )
        },
        Err(_) => copied_channel::<P>(image),
    }
}

unsafe extern "C-unwind" fn make_mat_channel_mut<P: PixelTypePrimitive + Pod>(
    image: &mut DynamicImageChannelView<'_>,
) {
    let mat = unsafe { mat_view(image) };
    if mat_buffer_is_exclusive(&mat) {
        // The pointer already points into this Mat's buffer; keep it.
        return;
    }

    // Assigning drops the previous channel, which releases its Mat
    *image = match mat.try_clone() {
        Ok(owned) => unsafe {
            // Safety: The cloned Mat keeps the buffer alive until the vtable drops it and the
            // vtable is built for `P`
            DynamicImageChannel::from_raw_parts(
                owned.data(),
                image.width(),
                image.height(),
                image.pixel_elements(),
                DynamicImageChannelView::raw_vtable(image),
                owned.into_raw().cast(),
            )
        },
        Err(_) => copied_channel::<P>(image),
    };
}

unsafe extern "C-unwind" fn drop_mat_channel(image: &mut DynamicImageChannelView<'_>) {
    drop(ManuallyDrop::into_inner(unsafe { mat_view(image) }));
}

fn copied_channel<P: PixelTypePrimitive + Pod>(
    image: &DynamicImageChannelView<'_>,
) -> DynamicImageChannel {
    // Safety: The vtable is built for P
    let data = unsafe { image.buffer_flat_unchecked::<P>() }.into();
    unsafe {
        DynamicImageChannel::from_arc_flat_unchecked(
            data,
            image.width(),
            image.height(),
            image.pixel_elements(),
        )
    }
}

fn nonzero_dimension(value: i32) -> opencv::Result<NonZeroU32> {
    u32::try_from(value)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            opencv::Error::new(Code::StsOutOfRange.into(), "Mat dimension must be positive")
        })
}

fn nonzero_pixel_elements(value: i32) -> opencv::Result<NonZeroU8> {
    u8::try_from(value)
        .ok()
        .and_then(NonZeroU8::new)
        .ok_or_else(|| {
            opencv::Error::new(
                Code::StsOutOfRange.into(),
                "Mat channels must be in 1..=255",
            )
        })
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::num::NonZeroU32;

    use imbuf::Image;
    use opencv::core::{Code, Mat, VecN};
    use opencv::traits::OpenCVIntoExternContainer;
    use testresult::TestResult;

    use super::*;
    use crate::ImbufAsOpencv;

    fn dimensions(width: u32, height: u32) -> (NonZeroU32, NonZeroU32) {
        (
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        )
    }

    #[test]
    fn gray_u8_shares_buffer() {
        let mat = Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]]).unwrap();
        let data_ptr = mat.data() as usize;
        let dynamic = mat.into_imbuf_dynamic().unwrap();

        assert_eq!(dynamic.len_nonzero().get(), 1);
        let channel = dynamic
            .first()
            .try_cast::<u8>()
            .expect("Expected U8 channel");
        assert_eq!(channel.pixel_elements().get(), 1);
        assert_eq!(channel.dimensions(), dimensions(3, 2));
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn gray_u16_shares_buffer() {
        let mat = Mat::from_slice_2d(&[[10u16, 20], [30, 40]]).unwrap();
        let data_ptr = mat.data() as usize;
        let dynamic = mat.into_imbuf_dynamic().unwrap();

        assert_eq!(dynamic.len_nonzero().get(), 1);
        let channel = dynamic
            .first()
            .try_cast::<u16>()
            .expect("Expected U16 channel");
        assert_eq!(channel.pixel_elements().get(), 1);
        assert_eq!(channel.dimensions(), dimensions(2, 2));
        assert_eq!(channel.buffer_flat(), &[10, 20, 30, 40]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn f32_shares_buffer() {
        let mat = Mat::from_slice_2d(&[[0.5f32, 1.5], [2.5, 3.5]]).unwrap();
        let data_ptr = mat.data() as usize;
        let dynamic = mat.into_imbuf_dynamic().unwrap();

        let channel = dynamic
            .first()
            .try_cast::<f32>()
            .expect("Expected F32 channel");
        assert_eq!(channel.pixel_elements().get(), 1);
        assert_eq!(channel.dimensions(), dimensions(2, 2));
        assert_eq!(channel.buffer_flat(), &[0.5, 1.5, 2.5, 3.5]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn interleaved_channels_are_a_single_dynamic_channel() -> TestResult {
        let mat = Mat::from_slice_2d(&[[VecN([1u8, 2, 3]), VecN([4, 5, 6])]]).unwrap();
        let dynamic = mat.into_imbuf_dynamic().unwrap();

        assert_eq!(dynamic.len_nonzero().get(), 1);
        assert_matches!(
            Image::<u8, 3>::try_from(dynamic.clone()),
            Err(imbuf::IncompatibleImage { .. })
        );
        let interleaved = Image::<[u8; 3], 1>::try_from(dynamic.clone())?;
        let [channel] = interleaved.into_channels();
        assert_eq!(channel.pixel_elements().get(), 3);
        assert_eq!(channel.dimensions(), dimensions(2, 1));
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4, 5, 6]);
        Ok(())
    }

    #[test]
    fn planar_mat_maps_to_planar_image() {
        let flat =
            Mat::from_slice_2d(&[[1u16, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]).unwrap();
        let mat: Mat = flat
            .reshape_nd(1, &[3, 2, 2])
            .unwrap()
            .opencv_into_extern_container_nofail();
        assert_eq!(mat.dims(), 3);

        let data_ptr = mat.data() as usize;
        let dynamic = mat.into_imbuf_dynamic().unwrap();
        assert!(dynamic.is_uniform());
        assert_eq!(dynamic.len_nonzero().get(), 3);

        let first = dynamic
            .first()
            .try_cast::<u16>()
            .expect("Expected U16 channel");
        assert_eq!(first.pixel_elements().get(), 1);
        assert_eq!(first.buffer_flat().as_ptr() as usize, data_ptr);

        let image = Image::<u16, 3>::try_from(dynamic).unwrap();
        assert_eq!(image.dimensions(), dimensions(2, 2));
        assert_eq!(
            image.buffers(),
            [
                &[1u16, 2, 3, 4][..],
                &[5, 6, 7, 8][..],
                &[9, 10, 11, 12][..]
            ]
        );
    }

    #[test]
    fn specific_conversion_matches_the_requested_type() {
        let mat = Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]]).unwrap();
        let image: Image<u8, 1> = mat.into_imbuf_specific().unwrap();
        assert_eq!(image.dimensions(), dimensions(3, 2));
        assert_eq!(image.buffer(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn clone_shares_buffer_and_survives_drop_of_original() {
        let mat = Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap();
        let data_ptr = mat.data() as usize;
        let dynamic = mat.into_imbuf_dynamic().unwrap();
        let cloned = dynamic.clone();
        drop(dynamic);

        let channel = cloned
            .first()
            .try_cast::<u8>()
            .expect("Expected U8 channel");
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn make_mut_copies_shared_data() {
        let mat = Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap();
        let data_ptr = mat.data() as usize;
        let mut dynamic = mat.into_imbuf_dynamic().unwrap();
        let cloned = dynamic.clone();

        let channel = dynamic[0]
            .try_cast_mut::<u8>()
            .expect("Expected U8 channel");
        channel.make_mut_flat()[0] = 42;

        let channel = dynamic[0].try_cast::<u8>().expect("Expected U8 channel");
        assert_eq!(channel.buffer_flat(), &[42, 2, 3, 4]);
        assert_ne!(channel.buffer_flat().as_ptr() as usize, data_ptr);

        let channel = cloned[0].try_cast::<u8>().expect("Expected U8 channel");
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn make_mut_reuses_unique_mat_buffer() {
        let mat = Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap();
        let data_ptr = mat.data() as usize;
        let mut dynamic = mat.into_imbuf_dynamic().unwrap();

        let channel = dynamic[0]
            .try_cast_mut::<u8>()
            .expect("Expected U8 channel");
        channel.make_mut_flat()[0] = 42;

        let channel = dynamic[0].try_cast::<u8>().expect("Expected U8 channel");
        assert_eq!(channel.buffer_flat(), &[42, 2, 3, 4]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn make_mut_copies_external_data_mat() {
        let backing = [1u8, 2, 3, 4];
        let borrowed = Mat::new_rows_cols_with_data(2, 2, &backing).unwrap();
        let mat = borrowed.opencv_into_extern_container_nofail();
        let data_ptr = mat.data() as usize;
        let mut dynamic = mat.into_imbuf_dynamic().unwrap();

        let channel = dynamic[0]
            .try_cast_mut::<u8>()
            .expect("Expected U8 channel");
        channel.make_mut_flat()[0] = 42;

        let channel = dynamic[0].try_cast::<u8>().expect("Expected U8 channel");
        assert_eq!(channel.buffer_flat(), &[42, 2, 3, 4]);
        assert_ne!(channel.buffer_flat().as_ptr() as usize, data_ptr);
        assert_eq!(backing, [1, 2, 3, 4]);
    }

    #[test]
    fn refcount_detects_shallow_mat_copies() {
        let mat = Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap();
        assert!(mat_buffer_is_exclusive(&mat));

        let shared = Mat::copy(&mat)
            .unwrap()
            .opencv_into_extern_container()
            .unwrap();
        assert!(!mat_buffer_is_exclusive(&mat));

        drop(shared);
        assert!(mat_buffer_is_exclusive(&mat));
    }

    #[test]
    fn make_mut_on_second_plane_copies_only_that_plane() {
        let flat =
            Mat::from_slice_2d(&[[1u16, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]).unwrap();
        let mat: Mat = flat
            .reshape_nd(1, &[3, 2, 2])
            .unwrap()
            .opencv_into_extern_container_nofail();
        let data_ptr = mat.data() as usize;
        let mut dynamic = mat.into_imbuf_dynamic().unwrap();
        let cloned = dynamic.clone();
        let plane_ptrs: Vec<usize> = (0..dynamic.len_nonzero().get())
            .map(|index| dynamic[index].buffer_flat_bytes().as_ptr() as usize)
            .collect();
        assert_eq!(plane_ptrs[0], data_ptr);

        let channel = dynamic[1]
            .try_cast_mut::<u16>()
            .expect("Expected U16 channel");
        channel.make_mut_flat()[0] = 42;

        let channel = dynamic[1].try_cast::<u16>().expect("Expected U16 channel");
        assert_eq!(channel.buffer_flat(), &[42, 6, 7, 8]);
        assert_ne!(channel.buffer_flat().as_ptr() as usize, plane_ptrs[1]);

        for index in [0usize, 2] {
            let channel = dynamic[index]
                .try_cast::<u16>()
                .expect("Expected U16 channel");
            assert_eq!(channel.buffer_flat().as_ptr() as usize, plane_ptrs[index]);
        }

        let channel = cloned[1].try_cast::<u16>().expect("Expected U16 channel");
        assert_eq!(channel.buffer_flat(), &[5, 6, 7, 8]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, plane_ptrs[1]);
    }

    #[test]
    fn round_trip_through_borrowed_mat() {
        let mat = Mat::from_slice_2d(&[[1u16, 2], [3, 4]]).unwrap();
        let dynamic = mat.into_imbuf_dynamic().unwrap();

        let borrowed = dynamic.as_opencv().unwrap();
        let mat = &borrowed;
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.data_typed::<u16>().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn empty_mat_is_rejected() {
        fn assert_2d(s: &[impl AsRef<[u8]>], desc: &str) {
            let x = Mat::from_slice_2d(s)
                .unwrap()
                .opencv_into_extern_container_nofail();
            assert_mat(x, desc);
        }
        fn assert_mat(mat: Mat, desc: &str) {
            let error = mat.into_imbuf_dynamic().expect_err("{desc}");
            assert_eq!(error.code_as_enum(), Some(Code::StsBadArg), "{desc}");
        }

        let empty_outer: &[[u8; 10]] = &[];
        let empty_inner_zero_i2: &[[u8; 0]] = &[[], []];
        let empty_inner_zero_i1: &[[u8; 0]] = &[[]];
        let empty_inner_and_outer: &[[u8; 0]] = &[];

        assert_mat(Mat::default(), "default");
        assert_2d(empty_outer, "empty_outer");
        assert_2d(empty_inner_zero_i2, "empty_inner_zero_i2");
        assert_2d(empty_inner_zero_i1, "empty_inner_zero_i1");
        assert_2d(empty_inner_and_outer, "empty_inner_and_outer");
    }

    #[test]
    fn unsupported_depth_is_rejected() {
        let s = &[[1i32, 2], [3, 4]];
        let mat = Mat::from_slice_2d(s).unwrap();
        let error = mat.into_imbuf_dynamic().unwrap_err();
        assert_eq!(error.code_as_enum(), Some(Code::StsUnsupportedFormat));
    }

    #[test]
    fn non_continuous_mat_is_rejected() {
        let mat = Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]]).unwrap();
        let column = mat.col(0).unwrap().opencv_into_extern_container_nofail();
        assert!(!column.is_continuous());

        let error = column.into_imbuf_dynamic().unwrap_err();
        assert_eq!(error.code_as_enum(), Some(Code::StsUnmatchedSizes));
    }
}
