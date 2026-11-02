use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    sync::atomic::{AtomicI32, Ordering},
};

use bytemuck::Pod;
use imbuf::{
    DynamicImage, DynamicImageChannel, DynamicImageChannelView, DynamicImageView,
    ImageChannelVTable, ImageView, IncompatibleImageError, PixelType, PixelTypePrimitive,
};
use opencv::{
    boxed_ref::{BoxedRef, BoxedRefMut},
    core::{
        CV_8U, CV_16U, CV_32F, CV_MAT_DEPTH_MASK, Code, Mat, MatTraitConst, Range, UMatData,
        UMatDataTraitConst, Vector,
    },
    prelude::MatTraitConstManual,
    traits::{Boxed, OpenCVIntoExternContainer},
};

/// Converts `OpenCV` matrices into [`imbuf`] images without copying the pixels. The image keeps
/// the buffer alive the way the source takes part in the reference counting of `OpenCV`:
///
/// - An owned [`Mat`] is moved into a [`DynamicImage`], which holds its reference.
/// - A [`BoxedRef<'a, Mat>`] (e.g. of `Mat::new_rows_cols_with_data` or `Mat::roi`) and a
///   `&'a Mat` become an image, which borrows the buffer for `'a`. Mutable access copies it.
/// - A [`BoxedRefMut<'a, Mat>`] and a `&'a mut Mat` become an image, which mutably borrows the
///   buffer for `'a`. Mutable access writes into it, clones copy it.
///
/// A borrowed image can't outlive the buffer:
///
/// ```compile_fail,E0597
/// use imbuf_opencv::OpencvIntoImbuf;
/// use opencv::core::Mat;
///
/// let image = {
///     let data = vec![0u8; 4];
///     Mat::new_rows_cols_with_data(2, 2, &data).unwrap().into_imbuf_dynamic().unwrap()
/// };
/// ```
pub trait OpencvIntoImbuf<'a>: Sized {
    /// # Errors
    /// Fails if the `Mat` is empty, not contiguous, not 2D/3D, has an
    /// unsupported depth or its data isn't aligned for its depth.
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImageView<'a>>;

    /// # Errors
    /// Fails for the same reasons as [`OpencvIntoImbuf::into_imbuf_dynamic`], or
    /// if the `Mat` layout does not match the requested [`imbuf::ImageView`] type.
    fn into_imbuf_specific<TP: PixelType, const CHANNELS: usize>(
        self,
    ) -> opencv::Result<ImageView<'a, TP, CHANNELS>>
    where
        ImageView<'a, TP, CHANNELS>: TryFrom<DynamicImageView<'a>, Error = IncompatibleImageError>,
    {
        ImageView::try_from(self.into_imbuf_dynamic()?)
            .map_err(|e| opencv::Error::new(Code::StsUnsupportedFormat.into(), e.to_string()))
    }
}

impl OpencvIntoImbuf<'static> for Mat {
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImage> {
        // Safety: An owned Mat keeps its buffer alive (external data has to outlive the Mat
        // by the contract of its unsafe constructor)
        unsafe { dynamic_image::<OwnedMatFactory>(self) }
    }
}

impl<'a> OpencvIntoImbuf<'a> for BoxedRef<'a, Mat> {
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImageView<'a>> {
        // The image borrows the buffer for 'a like the BoxedRef, so the Mat can be taken out
        let mat = self.opencv_into_extern_container_nofail();
        // Safety: The buffer is borrowed for 'a
        unsafe { dynamic_image::<BorrowedMatFactory>(mat) }
    }
}

impl<'a> OpencvIntoImbuf<'a> for BoxedRefMut<'a, Mat> {
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImageView<'a>> {
        // The image mutably borrows the buffer for 'a like the BoxedRefMut, so the Mat can be
        // taken out
        let mat = self.opencv_into_extern_container_nofail();
        // Safety: The buffer is exclusively borrowed for 'a
        unsafe { dynamic_image::<BorrowedMutMatFactory>(mat) }
    }
}

impl<'a> OpencvIntoImbuf<'a> for &'a Mat {
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImageView<'a>> {
        Mat::copy(self)?.into_imbuf_dynamic()
    }
}

impl<'a> OpencvIntoImbuf<'a> for &'a mut Mat {
    fn into_imbuf_dynamic(self) -> opencv::Result<DynamicImageView<'a>> {
        Mat::copy_mut(self)?.into_imbuf_dynamic()
    }
}

/// Creates the vtable for channels of the primitive `P`, which hold the `Mat` of their buffer in
/// `data`. The factory decides, how the channel holds the buffer.
trait MatChannelFactory<P> {
    const VTABLE: &'static ImageChannelVTable;
}

/// The `Mat` holds a reference of the buffer, which can be shared with other `Mat`s
struct OwnedMatFactory;

/// The buffer is borrowed for the lifetime of the channel
struct BorrowedMatFactory;

/// The buffer is exclusively borrowed for the lifetime of the channel
struct BorrowedMutMatFactory;

impl<P: PixelTypePrimitive + Pod> MatChannelFactory<P> for OwnedMatFactory {
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable {
        clone: share_mat_channel::<P>,
        make_mut: make_owned_mat_channel_mut::<P>,
        drop: drop_mat_channel,
        primitive_size: primitive_size::<P>,
        primitive_alignment: primitive_alignment::<P>,
    };
}

impl<P: PixelTypePrimitive + Pod> MatChannelFactory<P> for BorrowedMatFactory {
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable {
        clone: share_mat_channel::<P>,
        make_mut: make_borrowed_mat_channel_mut::<P>,
        drop: drop_mat_channel,
        primitive_size: primitive_size::<P>,
        primitive_alignment: primitive_alignment::<P>,
    };
}

impl<P: PixelTypePrimitive + Pod> MatChannelFactory<P> for BorrowedMutMatFactory {
    const VTABLE: &'static ImageChannelVTable = &ImageChannelVTable {
        clone: copy_mat_channel::<P>,
        make_mut: write_through_mat_channel,
        drop: drop_mat_channel,
        primitive_size: primitive_size::<P>,
        primitive_alignment: primitive_alignment::<P>,
    };
}

/// # Safety
/// The buffer of `mat` has to fit the factory `F` for `'a`: An owned `Mat` keeps it alive, a
/// borrowed one is valid for `'a` (and exclusively borrowed for [`BorrowedMutMatFactory`]).
unsafe fn dynamic_image<'a, F>(mat: Mat) -> opencv::Result<DynamicImageView<'a>>
where
    F: MatChannelFactory<u8> + MatChannelFactory<u16> + MatChannelFactory<f32>,
{
    // Safety: Forwarded to the caller
    unsafe {
        match mat.typ() & CV_MAT_DEPTH_MASK {
            CV_8U => typed_dynamic_image::<u8, F>(mat),
            CV_16U => typed_dynamic_image::<u16, F>(mat),
            CV_32F => typed_dynamic_image::<f32, F>(mat),
            _ => Err(opencv::Error::new(
                Code::StsUnsupportedFormat.into(),
                "Mat depth must be CV_8U, CV_16U or CV_32F",
            )),
        }
    }
}

/// # Safety
/// See [`dynamic_image`]
unsafe fn typed_dynamic_image<'a, P, F>(mat: Mat) -> opencv::Result<DynamicImageView<'a>>
where
    P: PixelTypePrimitive + Pod,
    F: MatChannelFactory<P>,
{
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
    // The vtable guarantees buffers, which are aligned for P. Mats allocated by OpenCV are,
    // but Mats over external data might not be. Planes start at multiples of size_of::<P>().
    if !mat.data().cast::<P>().is_aligned() {
        return Err(opencv::Error::new(
            Code::StsUnsupportedFormat.into(),
            "Mat data is not aligned for its depth",
        ));
    }

    if dims == 2 {
        let width = nonzero_dimension(mat.cols())?;
        let height = nonzero_dimension(mat.rows())?;
        let pixel_elements = nonzero_pixel_elements(mat.channels())?;
        // Safety: The Mat is checked above, its buffer is forwarded to the caller
        let channel = unsafe { into_channel::<P, F>(mat, width, height, pixel_elements) };
        Ok(DynamicImageView::from_channels(channel, []))
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
            .map(|plane| {
                // Safety: The planes of the checked Mat are disjoint parts of its buffer
                unsafe { planar_channel::<P, F>(&mat, i32::from(plane), width, height) }
            })
            .collect::<opencv::Result<Vec<_>>>()?;
        let mut channels = channels.into_iter();
        let Some(first) = channels.next() else {
            return Err(opencv::Error::new(
                Code::StsBadArg.into(),
                "Mat must have at least one plane",
            ));
        };
        Ok(DynamicImageView::from_channels(first, channels))
    }
}

/// # Safety
/// `mat` must be a continuous `Mat` of `height` x `width` pixels with `pixel_elements`
/// elements of `P` and be aligned for `P`. Its buffer has to fit the factory `F` for `'a`.
unsafe fn into_channel<'a, P, F>(
    mat: Mat,
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
) -> DynamicImageChannelView<'a>
where
    P: PixelTypePrimitive + Pod,
    F: MatChannelFactory<P>,
{
    let ptr = mat.data();
    let data = mat.into_raw().cast::<()>();
    // Safety: The Mat keeps the buffer alive until the vtable drops it (or it's borrowed for
    // 'a) and the vtable is built for `P`
    unsafe {
        DynamicImageChannelView::from_raw_parts(ptr, width, height, pixel_elements, F::VTABLE, data)
    }
}

/// Moves `mat`, which `Mat::try_clone` allocated, into a channel
fn owned_channel<P: PixelTypePrimitive + Pod>(
    mat: Mat,
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
) -> DynamicImageChannel {
    debug_assert!(has_refcount(&mat));
    // Planes of a 3D Mat stay 3D, so only the number of elements is checked
    debug_assert_eq!(
        Ok(()),
        DynamicImageChannelView::ensure_valid_flat_len(
            width,
            height,
            pixel_elements,
            mat.total() * usize::try_from(mat.channels()).unwrap_or(0)
        )
    );
    // Safety: A clone is a continuous, aligned Mat, which owns its buffer
    unsafe { into_channel::<P, OwnedMatFactory>(mat, width, height, pixel_elements) }
}

/// # Safety
/// See [`into_channel`]. `mat` must be 3D and continuous.
unsafe fn planar_channel<'a, P, F>(
    mat: &Mat,
    plane: i32,
    width: NonZeroU32,
    height: NonZeroU32,
) -> opencv::Result<DynamicImageChannelView<'a>>
where
    P: PixelTypePrimitive + Pod,
    F: MatChannelFactory<P>,
{
    let mut ranges = Vector::<Range>::new();
    ranges.push(Range::new(plane, plane + 1)?);
    ranges.push(Range::all()?);
    ranges.push(Range::all()?);
    let plane = Mat::ranges(mat, &ranges)?.opencv_into_extern_container()?;
    // Safety: Forwarded to the caller
    Ok(unsafe { into_channel::<P, F>(plane, width, height, NonZeroU8::MIN) })
}

/// Mirrors the prefix of `cv::Mat` in `OpenCV` 4 (`flags` to `u`).
///
/// The generated `cv_Mat_propU` shim dereferences `u` and therefore crashes for
/// Mats over external data (`u == null`), so read the public field directly.
/// [`mat_header`] checks the layout before `u` is used.
#[repr(C)]
#[derive(Clone, Copy)]
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

/// Mirrors the prefix of `cv::UMatData` in `OpenCV` 4 (`prevAllocator` to `size`), so the
/// reference counts can be loaded atomically. [`mat_buffer_is_exclusive`] checks the layout
/// before the counts are used.
#[repr(C)]
struct UMatDataHeader {
    prev_allocator: *const c_void,
    curr_allocator: *const c_void,
    urefcount: i32,
    refcount: i32,
    data: *mut u8,
    origdata: *mut u8,
    size: usize,
}

/// The header of `mat`. Only `OpenCV` 4 is supported (checked at compile time), and all mirrored
/// fields with an accessor have to match it.
///
/// # Panics
/// If the layout of `cv::Mat` doesn't match the mirror
fn mat_header(mat: &Mat) -> MatHeader {
    const {
        assert!(
            opencv::core::CV_VERSION_MAJOR == 4,
            "Only works for opencv4 for now"
        );
    };
    // Safety: In OpenCV 4, `cv::Mat` starts with the mirrored fields, which are always
    // initialized. The header is owned by `mat` and isn't changed concurrently.
    let header = unsafe { mat.as_raw().cast::<MatHeader>().read() };
    let matches = header.flags == mat.flags()
        && header.dims == mat.dims()
        && header.rows == mat.rows()
        && header.cols == mat.cols()
        && header.data.cast::<u8>().cast_const() == mat.data()
        && header.datastart.cast::<u8>() == mat.datastart()
        && header.dataend.cast::<u8>() == mat.dataend()
        && header.datalimit.cast::<u8>() == mat.datalimit();
    assert!(matches, "The layout of cv::Mat doesn't match OpenCV 4");
    header
}

/// Whether `mat` has a reference count (`u != null`), so it keeps its buffer alive
fn has_refcount(mat: &Mat) -> bool {
    !mat_header(mat).u.is_null()
}

/// Returns `true` if this `Mat` header is the sole owner of its pixel buffer.
///
/// Mats over external data (`u == null`) have no refcount and are never
/// considered exclusive. `urefcount`/`mapcount` guard against a `UMat` or
/// device mapping aliasing the same allocation. If the layout of `cv::UMatData`
/// doesn't match the mirror, the buffer is considered shared (and copied).
fn mat_buffer_is_exclusive(mat: &Mat) -> bool {
    let u_ptr = mat_header(mat).u;
    if u_ptr.is_null() {
        return false;
    }
    // `UMatData` is generated as an owning boxed type; wrap it so dropping the
    // view does not delete the allocation owned by the `Mat`.
    // Safety: The layout of `cv::Mat` is checked, so `u` points to the `UMatData` of the buffer,
    // which `mat` keeps alive
    let u = ManuallyDrop::new(unsafe { UMatData::from_raw(u_ptr) });
    let u_header = u_ptr.cast::<UMatDataHeader>();
    // Safety: In OpenCV 4, `cv::UMatData` starts with the mirrored fields. `data`, `origdata`
    // and `size` don't change while `mat` references the buffer, so they can be read and compared
    // with their accessors to check the layout. No reference to the whole `UMatData` is created,
    // as the counts change concurrently.
    let (data, origdata, size) = unsafe {
        (
            (&raw const (*u_header).data).read(),
            (&raw const (*u_header).origdata).read(),
            (&raw const (*u_header).size).read(),
        )
    };
    if data.cast_const() != u.data() || origdata.cast_const() != u.origdata() || size != u.size() {
        return false;
    }
    // Every `Mat` and `UMat`, which shares the buffer, holds a reference. `OpenCV` changes the
    // counts atomically (`CV_XADD`). Like `Arc::get_mut`, the acquire loads synchronize with the
    // release of other references, so their writes to the buffer happen before ours.
    // Safety: The counts are aligned `int`s of the checked layout, which `OpenCV` only changes
    // atomically
    let (refcount, urefcount) = unsafe {
        (
            AtomicI32::from_ptr(&raw mut (*u_header).refcount).load(Ordering::Acquire),
            AtomicI32::from_ptr(&raw mut (*u_header).urefcount).load(Ordering::Acquire),
        )
    };
    // Without other references, nothing maps the buffer concurrently
    refcount == 1 && urefcount == 0 && u.mapcount() == 0
}

/// Reborrows the `cv::Mat` owned by the channel without taking ownership.
unsafe fn mat_view(image: &DynamicImageChannelView<'_>) -> ManuallyDrop<Mat> {
    let data = DynamicImageChannelView::raw_data(image);
    ManuallyDrop::new(unsafe { Mat::from_raw(data.cast()) })
}

unsafe extern "C-unwind" fn primitive_size<P>(_: &DynamicImageChannelView<'_>) -> NonZeroUsize {
    const { NonZeroUsize::new(size_of::<P>()).expect("Primitives are not zero sized") }
}

unsafe extern "C-unwind" fn primitive_alignment<P>(
    _: &DynamicImageChannelView<'_>,
) -> NonZeroUsize {
    const { NonZeroUsize::new(align_of::<P>()).expect("Alignments are never zero") }
}

/// Shares the buffer with a new `Mat`, which keeps the vtable (owned or borrowed for `'a`)
unsafe extern "C-unwind" fn share_mat_channel<'a, P: PixelTypePrimitive + Pod>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    let mat = unsafe { mat_view(image) };
    match Mat::copy(&*mat).and_then(OpenCVIntoExternContainer::opencv_into_extern_container) {
        Ok(cloned) => unsafe {
            // Safety: The cloned Mat keeps the buffer alive until the vtable drops it (or it's
            // borrowed for 'a like the original) and the vtable is built for `P`
            DynamicImageChannelView::from_raw_parts(
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

/// Copies the buffer into a new `Mat`, which owns it
unsafe extern "C-unwind" fn copy_mat_channel<'a, P: PixelTypePrimitive + Pod>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    let mat = unsafe { mat_view(image) };
    match mat.try_clone() {
        Ok(owned) => {
            owned_channel::<P>(owned, image.width(), image.height(), image.pixel_elements())
        }
        Err(_) => copied_channel::<P>(image),
    }
}

unsafe extern "C-unwind" fn make_owned_mat_channel_mut<P: PixelTypePrimitive + Pod>(
    image: &mut DynamicImageChannelView<'_>,
) {
    let mat = unsafe { mat_view(image) };
    if mat_buffer_is_exclusive(&mat) {
        // The pointer already points into this Mat's buffer; keep it.
        return;
    }
    // Assigning drops the previous channel, which releases its Mat
    *image = unsafe { copy_mat_channel::<P>(image) };
}

/// A shared borrow is never written, so mutable access copies the buffer
unsafe extern "C-unwind" fn make_borrowed_mat_channel_mut<P: PixelTypePrimitive + Pod>(
    image: &mut DynamicImageChannelView<'_>,
) {
    // Assigning drops the previous channel, which releases its Mat
    *image = unsafe { copy_mat_channel::<P>(image) };
}

/// The buffer is exclusively borrowed for the lifetime of the channel, so it's written directly
unsafe extern "C-unwind" fn write_through_mat_channel(_: &mut DynamicImageChannelView<'_>) {}

unsafe extern "C-unwind" fn drop_mat_channel(image: &mut DynamicImageChannelView<'_>) {
    drop(ManuallyDrop::into_inner(unsafe { mat_view(image) }));
}

fn copied_channel<P: PixelTypePrimitive + Pod>(
    image: &DynamicImageChannelView<'_>,
) -> DynamicImageChannel {
    // Safety: The vtable is built for P
    let data = unsafe { image.buffer_flat_unchecked::<P>() }.into();
    unsafe {
        DynamicImageChannel::new_arc_flat_unchecked(
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
    use crate::{ImbufAsOpencv, ImbufAsOpencvMut};
    use opencv::prelude::MatTraitManual;

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
            Err(imbuf::IncompatibleImageError::ChannelCount(_))
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
    fn borrowed_mat_is_shared_until_mutable_access() -> TestResult {
        let backing = [1u8, 2, 3, 4];
        let mut image = Mat::new_rows_cols_with_data(2, 2, &backing)?.into_imbuf_dynamic()?;
        assert_eq!(image[0].buffer_flat::<u8>()?.as_ptr(), backing.as_ptr());
        let clone = image.clone();

        image[0].make_mut_flat::<u8>()?[0] = 42;
        assert_eq!(image[0].buffer_flat::<u8>()?, &[42, 2, 3, 4]);
        assert_ne!(image[0].buffer_flat::<u8>()?.as_ptr(), backing.as_ptr());
        assert_eq!(clone[0].buffer_flat::<u8>()?.as_ptr(), backing.as_ptr());
        assert_eq!(backing, [1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn mutably_borrowed_mat_writes_through() -> TestResult {
        let mut backing = [1u8, 2, 3, 4];
        let ptr = backing.as_ptr();
        {
            let mut image =
                Mat::new_rows_cols_with_data_mut(2, 2, &mut backing)?.into_imbuf_dynamic()?;
            let clone = image.clone();
            image[0].make_mut_flat::<u8>()?[0] = 42;
            assert_eq!(image[0].buffer_flat::<u8>()?.as_ptr(), ptr);
            assert_eq!(clone[0].buffer_flat::<u8>()?, &[1, 2, 3, 4]);
            assert_ne!(clone[0].buffer_flat::<u8>()?.as_ptr(), ptr);
        }
        assert_eq!(backing, [42, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn referenced_mats_are_borrowed() -> TestResult {
        let mut mat = Mat::from_slice_2d(&[[1u8, 2], [3, 4]])?;
        let ptr = mat.data();
        {
            let image = (&mat).into_imbuf_dynamic()?;
            assert_eq!(image[0].buffer_flat_bytes().as_ptr(), ptr);
        }
        {
            let mut image = (&mut mat).into_imbuf_dynamic()?;
            image[0].make_mut_flat::<u8>()?[3] = 42;
        }
        assert_eq!(mat.data_typed::<u8>()?, &[1, 2, 3, 42]);
        assert_eq!(mat.data(), ptr);
        Ok(())
    }

    #[test]
    fn mutably_borrowed_planar_mat_writes_into_its_planes() -> TestResult {
        let flat = Mat::from_slice_2d(&[[1u16, 2], [3, 4], [5, 6], [7, 8]])?;
        let mut mat = flat
            .reshape_nd(1, &[2, 2, 2])?
            .opencv_into_extern_container_nofail();
        {
            let mut image = (&mut mat).into_imbuf_dynamic()?;
            image[1].make_mut_flat::<u16>()?[0] = 42;
        }
        let flat = mat.reshape_nd(1, &[8])?;
        assert_eq!(flat.data_typed::<u16>()?, &[1, 2, 3, 4, 42, 6, 7, 8]);
        Ok(())
    }

    #[test]
    fn mutable_view_of_mat_channel_keeps_it_exclusive() -> TestResult {
        let mut image = Mat::from_slice_2d(&[[1u8, 2], [3, 4]])?.into_imbuf_dynamic()?;
        let ptr = DynamicImageChannelView::raw_ptr(&image[0]);
        {
            let mut view = image.as_opencv_mut()?;
            view.data_typed_mut::<u8>()?[0] = 42;
        }
        // The view released its reference, so the buffer is still exclusive
        image[0].make_mut_flat::<u8>()?[1] = 43;
        assert_eq!(image[0].buffer_flat::<u8>()?, &[42, 43, 3, 4]);
        assert_eq!(DynamicImageChannelView::raw_ptr(&image[0]), ptr);
        Ok(())
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
    fn mat_header_matches_the_layout_of_opencv() {
        let owned = Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap();
        let header = mat_header(&owned);
        assert!(!header.u.is_null());

        let backing = [1u8, 2, 3, 4];
        let borrowed = Mat::new_rows_cols_with_data(2, 2, &backing).unwrap();
        // `backing` outlives the `Mat`
        let external = borrowed.opencv_into_extern_container_nofail();
        let header = mat_header(&external);
        assert!(header.u.is_null());
        assert!(!mat_buffer_is_exclusive(&external));
    }

    #[test]
    fn umat_of_the_buffer_is_detected() {
        let mat = Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap();
        let umat = mat
            .get_umat(
                opencv::core::AccessFlag::ACCESS_READ,
                opencv::core::UMatUsageFlags::USAGE_DEFAULT,
            )
            .unwrap();
        assert!(!mat_buffer_is_exclusive(&mat));

        drop(umat);
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
