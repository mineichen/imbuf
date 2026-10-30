use std::{
    fmt::{self, Debug, Formatter},
    iter::{self, once},
    marker::PhantomData,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    ops::Deref,
    slice, vec,
};

use bytemuck::Pod;

use crate::{
    ImageChannelView, ImageView, IncompatibleImage, InvalidLenError, PixelType, PixelTypePrimitive,
    channel::ImageChannelVTable,
    error::{
        IncompatibleChannelCount, IncompatibleChannelError, IncompatibleImageError,
        OverflowBufferLenError, PixelLayout,
    },
};
/// Image with number of channels and their types and dimensions only known at runtime
/// There are no guarantees that the types or dimensions of channels match. See [`ImageView`] for more information.
/// The public interface is designed, so it can be extended to support images, which cannot be represented with Image (e.g. 1 Channel u8 and the other f32)
///
/// The buffers are valid for the lifetime `'a`, see [`DynamicImageChannelView`].
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicImageView<'a> {
    channels: Vec<DynamicImageChannelView<'a>>,
}

/// [`DynamicImageView`] which owns its buffers (or shares them with other owners)
pub type DynamicImage = DynamicImageView<'static>;

impl<'a> DynamicImageView<'a> {
    pub fn from_channels(
        first: DynamicImageChannelView<'a>,
        rest: impl IntoIterator<Item = DynamicImageChannelView<'a>>,
    ) -> Self {
        Self {
            channels: once(first).chain(rest).collect(),
        }
    }
    /// `DynamicImageView` always has at least one channel, so this never panics
    #[must_use]
    pub fn first(&self) -> &DynamicImageChannelView<'a> {
        &self.channels[0]
    }

    /// `DynamicImageView` always has at least one channel, so this never panics
    #[must_use]
    pub fn last(&self) -> &DynamicImageChannelView<'a> {
        &self.channels[self.channels.len() - 1]
    }

    /// Checks, if all channels have the same width, height and [`PixelLayout`]
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        let (first, mut rest) = self.mapped_channels(|x| (x.pixel_layout(), x.dimensions()));
        rest.all(|x| x == first)
    }

    // Returns the first channel instance mapped plus a iterator of the mapped rest
    // This allows e.g. `Iterator::fold` without unwrapping
    #[allow(clippy::missing_panics_doc)]
    pub fn mapped_channels<'b, T: 'b, TF>(
        &'b self,
        mapper: TF,
    ) -> (
        T,
        iter::Map<slice::Iter<'b, DynamicImageChannelView<'a>>, TF>,
    )
    where
        TF: FnMut(&'b DynamicImageChannelView<'a>) -> T,
    {
        let mut iter = self.iter().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn mapped_channels_mut<'b, T: 'b, TF>(
        &'b mut self,
        mapper: TF,
    ) -> (
        T,
        iter::Map<slice::IterMut<'b, DynamicImageChannelView<'a>>, TF>,
    )
    where
        TF: FnMut(&'b mut DynamicImageChannelView<'a>) -> T,
    {
        let mut iter = self.iter_mut().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn into_mapped_channels<T, TF>(
        self,
        mapper: TF,
    ) -> (T, iter::Map<vec::IntoIter<DynamicImageChannelView<'a>>, TF>)
    where
        TF: FnMut(DynamicImageChannelView<'a>) -> T,
    {
        let mut iter = self.into_iter().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn len_nonzero(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.channels.len()).unwrap()
    }

    /// Checks, that the image consists of a single (e.g. interleaved) channel, as most other image
    /// libraries require
    ///
    /// # Errors
    /// [`IncompatibleChannelCount`], if the image has more than one channel
    pub fn ensure_single_channel(&self) -> Result<(), IncompatibleChannelCount> {
        let actual = self.len_nonzero();
        if actual == NonZeroUsize::MIN {
            Ok(())
        } else {
            Err(IncompatibleChannelCount {
                expected: NonZeroU8::MIN,
                actual,
            })
        }
    }

    /// Try to view this image as a `ImageView<T, CHANNELS>` without copying.
    ///
    /// Fails, if the channel count, the pixel layouts or the dimensions don't match.
    /// Use [`ImageView::view`] on the result to get a borrowed image by value.
    pub fn try_cast<T: PixelType, const CHANNELS: usize>(
        &self,
    ) -> Result<&ImageView<'a, T, CHANNELS>, IncompatibleImageError>
    where
        T::Primitive: Pod,
    {
        self.try_into()
    }

    /// Mutable version of [`DynamicImageView::try_cast`].
    pub fn try_cast_mut<T: PixelType, const CHANNELS: usize>(
        &mut self,
    ) -> Result<&mut ImageView<'a, T, CHANNELS>, IncompatibleImageError>
    where
        T::Primitive: Pod,
    {
        self.try_into()
    }

    /// Converts this image into a `ImageView<T, CHANNELS>` without copying.
    ///
    /// Drops the image on failure. To get it back, convert into
    /// `Result<ImageView, IncompatibleImage<DynamicImageView>>` with `From`/`Into` instead.
    pub fn try_into_typed<T: PixelType, const CHANNELS: usize>(
        self,
    ) -> Result<ImageView<'a, T, CHANNELS>, IncompatibleImageError>
    where
        T::Primitive: Pod,
    {
        self.try_into()
    }

    /// Checks, that the channel count, the pixel layouts and the dimensions match
    fn ensure_compatible<T: PixelType, const CHANNELS: usize>(
        &self,
    ) -> Result<(), IncompatibleImageError> {
        let actual = self.len_nonzero();
        if actual.get() != CHANNELS {
            return Err(IncompatibleChannelCount {
                expected: const { crate::unwrap_usize_to_nonzero_u8(CHANNELS) },
                actual,
            }
            .into());
        }
        self.channels
            .iter()
            .try_for_each(DynamicImageChannelView::ensure_fits_pixel::<T>)?;
        crate::image::check_dimensions(
            self.channels
                .iter()
                .map(DynamicImageChannelView::dimensions),
        )
    }
}

// Deref slice only, to make sure, noone can create a DynamicImageView with a empty Vec
impl<'a> Deref for DynamicImageView<'a> {
    type Target = [DynamicImageChannelView<'a>];

    fn deref(&self) -> &Self::Target {
        &self.channels
    }
}

impl<'a> IntoIterator for DynamicImageView<'a> {
    type Item = DynamicImageChannelView<'a>;
    type IntoIter = vec::IntoIter<DynamicImageChannelView<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.channels.into_iter()
    }
}

impl std::ops::DerefMut for DynamicImageView<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.channels
    }
}

/// Channel whose primitive and number of pixel elements are only known at runtime.
///
/// The channel only knows the size of its primitive (e.g. 2 bytes for `u16`). Its buffer can
/// therefore be viewed as any [`Pod`] primitive of the same size and alignment, e.g. via
/// [`DynamicImageChannelView::buffer_flat`], [`DynamicImageChannelView::try_cast`] or `ImageChannelView::try_from`.
/// For the same reason, only [`ImageChannelView`]s of `Pod` primitives can be converted into a `DynamicImageChannelView`:
///
/// ```compile_fail,E0277
/// use std::num::{NonZeroU8, NonZeroU32};
/// use imbuf::{DynamicImageChannel, ImageChannel, PixelTypePrimitive};
///
/// #[derive(Clone, Copy, PartialEq)]
/// struct NonZero(NonZeroU8);
/// impl PixelTypePrimitive for NonZero {}
///
/// let channel = ImageChannel::new_vec(vec![NonZero(NonZeroU8::MIN)], NonZeroU32::MIN, NonZeroU32::MIN).unwrap();
/// // Would allow writing 0 through `ImageChannel<u8>`
/// let dynamic = DynamicImageChannel::from(channel);
/// ```
///
/// The buffer is valid for the lifetime `'a`. Channels which own their buffer are `'static`
/// (see [`DynamicImageChannel`]). Borrowed channels can be created from borrowed typed channels
/// (e.g. [`ImageChannelView::view`]) or with [`DynamicImageChannelView::from_raw_parts`].
///
/// The raw parts are exposed as fields to `#[repr(C)]`, so a channel can be stored in and
/// retrieved from FFI types opaquely. The lifetime `'a` has to be re-established when reading
/// the fields back into a channel.
#[repr(C)]
pub struct DynamicImageChannelView<'a> {
    /// Pointer to the buffer, which is `len_flat()` primitives of the size
    /// `vtable.bytes_per_primitive()` long
    ptr: *const u8,
    width: NonZeroU32,
    height: NonZeroU32,
    vtable: &'static ImageChannelVTable,
    /// Storage specific data, which has to be cleaned up by `vtable.drop`
    data: *mut (),
    pixel_elements: NonZeroU8,
    lifetime: PhantomData<&'a ()>,
}

/// [`DynamicImageChannelView`] which owns its buffer (or shares it with other owners)
pub type DynamicImageChannel = DynamicImageChannelView<'static>;

impl<'a> DynamicImageChannelView<'a> {
    /// Creates a channel from its raw parts, e.g. to support a custom storage.
    ///
    /// # Safety
    /// The following invariants have to hold:
    /// - The size of the buffer in bytes (`width * height * pixel_elements * vtable.bytes_per_primitive()`)
    ///   doesn't exceed `usize::MAX`, and the multiplications to calculate it don't overflow
    ///   `usize`, which implies that `width * height * pixel_elements` matches the buffer length.
    ///   [`DynamicImageChannelView::ensure_valid_flat_len`] checks this, e.g. for untrusted input.
    /// - `ptr` points to an initialized buffer of that size, which stays valid until `vtable.drop`
    ///   is called and at least for `'a`. The buffer can be shared between instances.
    /// - `vtable` is able to work with `ptr` and `data` (see [`ImageChannelVTable`](crate::ImageChannelVTable)).
    ///   Any bytes written through a `Pod` view of the buffer have to be valid for the vtable
    ///   (e.g. the primitive of the vtable is `Pod` itself).
    /// - `ptr` and `data` can be sent to and shared between threads.
    ///
    /// # Panics
    /// In debug builds, panics if `width * height * pixel_elements` or the buffer size in bytes
    /// overflows `usize`.
    #[must_use]
    pub unsafe fn from_raw_parts(
        ptr: *const u8,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        vtable: &'static ImageChannelVTable,
        data: *mut (),
    ) -> Self {
        debug_assert_eq!(
            Ok(()),
            Self::ensure_valid_byte_len(
                width,
                height,
                pixel_elements,
                (vtable.bytes_per_primitive)()
            ),
            "The buffer size of the raw parts overflows usize"
        );
        Self {
            ptr,
            width,
            height,
            vtable,
            data,
            pixel_elements,
            lifetime: PhantomData,
        }
    }

    /// Read access to the buffer pointer, e.g. to support a custom storage or implement
    /// [`ImageChannelVTable`](crate::ImageChannelVTable).
    #[must_use]
    pub const fn raw_ptr(channel: &Self) -> *const u8 {
        channel.ptr
    }

    /// Read access to the storage specific data, e.g. to support a custom storage.
    #[must_use]
    pub const fn raw_data(channel: &Self) -> *mut () {
        channel.data
    }

    /// Read access to the vtable, e.g. to support a custom storage.
    #[must_use]
    pub const fn raw_vtable(channel: &Self) -> &'static ImageChannelVTable {
        channel.vtable
    }

    #[must_use]
    pub const fn width(&self) -> NonZeroU32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> NonZeroU32 {
        self.height
    }

    #[must_use]
    pub const fn dimensions(&self) -> (NonZeroU32, NonZeroU32) {
        (self.width, self.height)
    }

    #[must_use]
    pub const fn dimensions_zeroable(&self) -> (u32, u32) {
        (self.width.get(), self.height.get())
    }

    #[must_use]
    pub const fn pixel_elements(&self) -> NonZeroU8 {
        self.pixel_elements
    }

    /// Size of a single primitive in bytes
    #[must_use]
    pub fn bytes_per_primitive(&self) -> NonZeroUsize {
        (self.vtable.bytes_per_primitive)()
    }

    /// `pixel_elements` and `bytes_per_primitive` of the channel
    #[must_use]
    pub fn pixel_layout(&self) -> PixelLayout {
        PixelLayout::new(self.pixel_elements, self.bytes_per_primitive())
    }

    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "Overflow checked during construction"
        )]
        let width = self.width.get() as usize;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "Overflow checked during construction"
        )]
        let height = self.height.get() as usize;
        width * height
    }

    /// Number of primitives in the buffer (`width * height * pixel_elements`)
    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub const fn len_flat(&self) -> usize {
        self.len() * self.pixel_elements.get() as usize
    }

    /// Fails, if `pixel_elements`, the size or the alignment of `T::Primitive` don't match
    pub fn buffer<T: PixelType + Pod>(&self) -> Result<&[T], IncompatibleChannelError> {
        self.ensure_fits_pixel::<T>()?;
        // Safety: The layout is checked. Pod accepts any bit pattern
        Ok(unsafe { self.buffer_unchecked() })
    }

    /// # Safety
    /// `T` must have the layout of the channel's pixel (see [`DynamicImageChannelView::buffer`]) and the buffer must contain valid values of `T`.
    #[must_use]
    pub unsafe fn buffer_unchecked<T: PixelType>(&self) -> &[T] {
        debug_assert_eq!(Ok(()), self.ensure_fits_pixel::<T>());
        unsafe { slice::from_raw_parts(self.ptr.cast::<T>(), self.len()) }
    }

    /// Fails, if the size or the alignment of `T` don't match the primitive of the channel
    pub fn buffer_flat<T: PixelTypePrimitive + Pod>(
        &self,
    ) -> Result<&[T], IncompatibleChannelError> {
        self.ensure_fits_primitive::<T>()?;
        // Safety: Size and alignment are checked. Pod accepts any bit pattern
        Ok(unsafe { self.buffer_flat_unchecked() })
    }

    /// # Safety
    /// `T` must have the size of the channels primitive, `ptr` must be aligned for `T` and the
    /// buffer must contain valid values of `T`.
    #[must_use]
    pub unsafe fn buffer_flat_unchecked<T: PixelTypePrimitive>(&self) -> &[T] {
        debug_assert_eq!(Ok(()), self.ensure_fits_primitive::<T>());
        // Safety: Forwarded to the caller. The length is valid due to the invariants of the channel
        unsafe { slice::from_raw_parts(self.ptr.cast::<T>(), self.len_flat()) }
    }

    /// Mutable access to the buffer, which is copied if it is shared (copy on write).
    ///
    /// Fails, if the size or the alignment of `T` don't match the primitive of the channel
    pub fn make_mut_flat<T: PixelTypePrimitive + Pod>(
        &mut self,
    ) -> Result<&mut [T], IncompatibleChannelError> {
        self.ensure_fits_primitive::<T>()?;
        // Safety: Size and alignment are checked. Pod accepts any bit pattern
        Ok(unsafe { self.make_mut_flat_unchecked() })
    }

    pub(crate) unsafe fn make_mut_flat_unchecked<T: PixelTypePrimitive>(&mut self) -> &mut [T] {
        self.make_exclusive();
        // Safety: Size and alignment are checked after make_mut, which might reallocate
        debug_assert_eq!(Ok(()), self.ensure_fits_primitive::<T>());
        // Safety: Forwarded to the caller
        unsafe { slice::from_raw_parts_mut(self.ptr.cast::<T>().cast_mut(), self.len_flat()) }
    }

    /// Calls `vtable.make_mut`, so the buffer is exclusively owned by the channel afterwards
    pub(crate) fn make_exclusive(&mut self) {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.vtable.make_mut)(self) };
    }

    #[must_use]
    pub fn buffer_flat_bytes(&self) -> &[u8] {
        let len = self.len_flat() * self.bytes_per_primitive().get();
        // Safety: The buffer is initialized (invariant of the channel)
        unsafe { slice::from_raw_parts(self.ptr, len) }
    }

    /// Try to view this channel as a `ImageChannelView<TP>`.
    ///
    /// Fails, if `pixel_elements`, the size or the alignment of `TP::Primitive` don't match.
    /// Use [`ImageChannelView::view`] on the result to get a borrowed channel by value.
    pub fn try_cast<TP: PixelType>(
        &self,
    ) -> Result<&ImageChannelView<'a, TP>, IncompatibleChannelError>
    where
        TP::Primitive: Pod,
    {
        self.try_into()
    }

    /// Mutable version of [`DynamicImageChannelView::try_cast`].
    pub fn try_cast_mut<TP: PixelType>(
        &mut self,
    ) -> Result<&mut ImageChannelView<'a, TP>, IncompatibleChannelError>
    where
        TP::Primitive: Pod,
    {
        self.try_into()
    }

    pub fn try_into_typed<TP: PixelType>(
        self,
    ) -> Result<ImageChannelView<'a, TP>, IncompatibleChannelError>
    where
        TP::Primitive: Pod,
    {
        self.try_into()
    }

    /// Checks that `width * height * pixel_elements` doesn't overflow `usize` and matches
    /// `buffer_len`, e.g. to validate untrusted dimensions before passing them to
    /// [`DynamicImageChannelView::from_raw_parts`]. For buffers, which store one element per
    /// pixel, [`DynamicImageChannelView::ensure_valid_len`] can be used instead.
    pub const fn ensure_valid_flat_len(
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        buffer_len: usize,
    ) -> Result<(), InvalidLenError> {
        Self::ensure_len(width, height, pixel_elements, NonZeroU8::MIN, buffer_len)
    }

    /// Checks that `width * height` (the number of pixels) doesn't overflow `usize` and matches
    /// `buffer_len`, e.g. to validate untrusted dimensions for a buffer, which stores one element
    /// per pixel (like the buffers of [`ImageChannel::new_vec`](crate::ImageChannel::new_vec)). For buffers, which store
    /// multiple primitives per pixel, use [`DynamicImageChannelView::ensure_valid_flat_len`].
    pub const fn ensure_valid_len(
        width: NonZeroU32,
        height: NonZeroU32,
        buffer_len: usize,
    ) -> Result<(), InvalidLenError> {
        Self::ensure_len(width, height, NonZeroU8::MIN, NonZeroU8::MIN, buffer_len)
    }

    /// Checks that `width * height * pixel_elements * channels` doesn't overflow `usize` and
    /// matches `len`
    pub(crate) const fn ensure_len(
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        channels: NonZeroU8,
        len: usize,
    ) -> Result<(), InvalidLenError> {
        match Self::flat_len_checked(width, height, pixel_elements, channels) {
            Ok(expected) if expected == len => Ok(()),
            Ok(expected) => Err(InvalidLenError::BufferMismatch {
                expected,
                actual: len,
            }),
            Err(e) => Err(InvalidLenError::Overflow(e)),
        }
    }

    /// Fails with the reason, if the buffer can't be viewed as `TP`
    pub fn ensure_fits_pixel<TP: PixelType>(&self) -> Result<(), IncompatibleChannelError> {
        if self.pixel_elements() == TP::ELEMENTS
            && self.ensure_fits_primitive::<TP::Primitive>().is_ok()
        {
            Ok(())
        } else {
            let required = PixelLayout::of::<TP>();
            Err(self.incompatible_error::<TP, TP::Primitive>(required))
        }
    }

    /// Fails with the reason, if the flat buffer can't be viewed as `T`. In contrast to
    /// [`DynamicImageChannelView::ensure_fits_pixel`], `pixel_elements` doesn't have to match.
    pub(crate) fn ensure_fits_primitive<T: PixelTypePrimitive>(
        &self,
    ) -> Result<(), IncompatibleChannelError> {
        if self.bytes_per_primitive().get() == size_of::<T>() && self.ptr.cast::<T>().is_aligned() {
            Ok(())
        } else {
            let required = PixelLayout::of_primitive::<T>(self.pixel_elements());
            Err(self.incompatible_error::<T, T>(required))
        }
    }

    /// Describes, why the buffer can't be viewed as `TTarget` with the primitive `TPrimitive`
    fn incompatible_error<TTarget, TPrimitive>(
        &self,
        required: PixelLayout,
    ) -> IncompatibleChannelError {
        IncompatibleChannelError {
            actual: self.pixel_layout(),
            required: Some(required),
            aligned: self.ptr.cast::<TPrimitive>().is_aligned(),
            target_pixel_type: std::any::type_name::<TTarget>(),
        }
    }

    /// Checks that the size of the buffer in bytes (`width * height * pixel_elements *
    /// bytes_per_primitive`) doesn't overflow `usize`, which [`DynamicImageChannelView::from_raw_parts`]
    /// requires
    fn ensure_valid_byte_len(
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        bytes_per_primitive: NonZeroUsize,
    ) -> Result<(), OverflowBufferLenError> {
        let len = Self::flat_len_checked(width, height, pixel_elements, NonZeroU8::MIN)?;
        len.checked_mul(bytes_per_primitive.get())
            .map(drop)
            .ok_or(OverflowBufferLenError::new(
                width,
                height,
                pixel_elements,
                NonZeroU8::MIN,
            ))
    }

    /// `width * height * pixel_elements * channels` as `usize`
    ///
    /// The calculation is done in `u64`, so it cannot truncate on targets with a smaller `usize`
    /// (see the assertion after this `impl`). The product of the two `u32` dimensions always fits
    /// `u64`, only the multiplication with `pixel_elements * channels` can overflow it.
    pub(crate) const fn flat_len_checked(
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        channels: NonZeroU8,
    ) -> Result<usize, OverflowBufferLenError> {
        #[allow(clippy::cast_lossless)]
        let pixels = (width.get() as u64) * (height.get() as u64);
        #[allow(clippy::cast_lossless)]
        let elements_per_pixel = (pixel_elements.get() as u64) * (channels.get() as u64);
        match pixels.checked_mul(elements_per_pixel) {
            Some(len) if len <= usize::MAX as u64 => {
                // The value is checked above to fit into `usize`
                #[allow(clippy::cast_possible_truncation)]
                Ok(len as usize)
            }
            _ => Err(OverflowBufferLenError::new(
                width,
                height,
                pixel_elements,
                channels,
            )),
        }
    }
}

// `usize as u64` is only lossless, if `usize` doesn't exceed 8 bytes
const _: () = assert!(size_of::<usize>() <= size_of::<u64>());

// Borrowed channels have to be dropped while 'a is alive, as the vtable might access the buffer
impl Drop for DynamicImageChannelView<'_> {
    fn drop(&mut self) {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.vtable.drop)(self) };
    }
}

impl Clone for DynamicImageChannelView<'_> {
    fn clone(&self) -> Self {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.vtable.clone)(self) }
    }
}

/// Channels are equal, if their dimensions, `pixel_elements`, `bytes_per_primitive` and bytes are equal
impl PartialEq for DynamicImageChannelView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions() == other.dimensions()
            && self.pixel_elements() == other.pixel_elements()
            && self.bytes_per_primitive() == other.bytes_per_primitive()
            && self.buffer_flat_bytes() == other.buffer_flat_bytes()
    }
}

impl Debug for DynamicImageChannelView<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicImageChannel")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("bytes_per_primitive", &self.bytes_per_primitive())
            .field("pixel_elements", &self.pixel_elements())
            .finish()
    }
}

// Safety: The primitives are Send + Sync and storages have to be thread safe (invariant of `from_raw_parts`)
unsafe impl Send for DynamicImageChannelView<'_> {}
unsafe impl Sync for DynamicImageChannelView<'_> {}

impl<'a, TPixel: PixelType, const CHANNELS: usize> From<ImageView<'a, TPixel, CHANNELS>>
    for DynamicImageView<'a>
where
    TPixel::Primitive: Pod,
{
    fn from(value: ImageView<'a, TPixel, CHANNELS>) -> Self {
        DynamicImageView {
            channels: <[ImageChannelView<'a, TPixel>; CHANNELS]>::from(value)
                .into_iter()
                .map(ImageChannelView::into)
                .collect(),
        }
    }
}

/// Hands back the input in [`IncompatibleImage`], if it can't be converted. Use `TryFrom`, if the
/// input isn't needed on failure.
impl<'a, T: PixelType, const CHANNELS: usize> From<DynamicImageView<'a>>
    for Result<ImageView<'a, T, CHANNELS>, IncompatibleImage<DynamicImageView<'a>>>
where
    T::Primitive: Pod,
{
    fn from(value: DynamicImageView<'a>) -> Self {
        if let Err(error) = value.ensure_compatible::<T, CHANNELS>() {
            return Err(IncompatibleImage {
                image: value,
                error,
            });
        }
        let mut iter = value.channels.into_iter();
        let channels = std::array::from_fn(|_| {
            let channel = iter.next().expect("The channel count is checked above");
            // Safety: Size, alignment and pixel_elements are checked above. Pod accepts any bit pattern
            unsafe { ImageChannelView::from_dynamic_unchecked(channel) }
        });
        Ok(ImageView::from_channels_unchecked(channels))
    }
}

/// Drops the input on failure. To get it back, convert into
/// `Result<ImageView, IncompatibleImage<DynamicImageView>>` with `From`/`Into` instead:
///
/// ```
/// use std::num::NonZeroU32;
/// use imbuf::{DynamicImage, Image, IncompatibleImage};
///
/// let luma = Image::<u8, 1>::new_vec(vec![42], NonZeroU32::MIN, NonZeroU32::MIN).unwrap();
/// let result: Result<Image<u16, 1>, IncompatibleImage<DynamicImage>> =
///     DynamicImage::from(luma).into();
/// let IncompatibleImage { image, error, .. } = result.unwrap_err();
/// assert_eq!(Image::<u8, 1>::try_from(image).unwrap().buffer(), &[42]);
/// println!("{error}");
/// ```
impl<'a, T: PixelType, const CHANNELS: usize> TryFrom<DynamicImageView<'a>>
    for ImageView<'a, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError;

    fn try_from(value: DynamicImageView<'a>) -> Result<Self, Self::Error> {
        Result::<Self, IncompatibleImage<_>>::from(value).map_err(|e| e.error)
    }
}

/// Borrows the channels without copying. Mutable access (e.g. [`ImageView::make_mut`]) copies
/// the buffers, the `DynamicImageView` stays unchanged.
impl<'a, 'b, T: PixelType, const CHANNELS: usize> TryFrom<&'b DynamicImageView<'a>>
    for ImageView<'b, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError;

    fn try_from(value: &'b DynamicImageView<'a>) -> Result<Self, Self::Error> {
        <&ImageView<'a, T, CHANNELS>>::try_from(value).map(ImageView::view)
    }
}

/// Mutably borrows the channels without copying. Mutable access (e.g. [`ImageView::make_mut`])
/// writes into the buffers of the `DynamicImageView`.
impl<'a, 'b, T: PixelType, const CHANNELS: usize> TryFrom<&'b mut DynamicImageView<'a>>
    for ImageView<'b, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError;

    fn try_from(value: &'b mut DynamicImageView<'a>) -> Result<Self, Self::Error> {
        <&mut ImageView<'a, T, CHANNELS>>::try_from(value).map(ImageView::view_mut)
    }
}

/// Views the channels as an image without copying, see [`DynamicImageView::try_cast`]
impl<'a, 'b, T: PixelType, const CHANNELS: usize> TryFrom<&'b DynamicImageView<'a>>
    for &'b ImageView<'a, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError;

    fn try_from(value: &'b DynamicImageView<'a>) -> Result<Self, Self::Error> {
        value.ensure_compatible::<T, CHANNELS>()?;
        let channels = value
            .channels
            .as_slice()
            .try_into()
            .expect("The channel count is checked above");
        // Safety: Size, alignment, pixel_elements and dimensions are checked above. Pod accepts
        // any bit pattern
        Ok(unsafe { ImageView::from_dynamic_ref_unchecked(channels) })
    }
}

/// Mutably views the channels as an image without copying, see [`DynamicImageView::try_cast_mut`]
impl<'a, 'b, T: PixelType, const CHANNELS: usize> TryFrom<&'b mut DynamicImageView<'a>>
    for &'b mut ImageView<'a, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError;

    fn try_from(value: &'b mut DynamicImageView<'a>) -> Result<Self, Self::Error> {
        value.ensure_compatible::<T, CHANNELS>()?;
        let channels = value
            .channels
            .as_mut_slice()
            .try_into()
            .expect("The channel count is checked above");
        // Safety: Size, alignment, pixel_elements and dimensions are checked above. Pod accepts
        // any bit pattern
        Ok(unsafe { ImageView::from_dynamic_mut_unchecked(channels) })
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::num::{NonZeroU8, NonZeroU32};

    use testresult::TestResult;

    use super::*;
    use crate::{Image, ImageChannel, ImageView};

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    const NZ_3: NonZeroU32 = NonZeroU32::new(3).unwrap();
    const NZ_U8_2: NonZeroU8 = NonZeroU8::new(2).unwrap();
    const NZ_U8_3: NonZeroU8 = NonZeroU8::new(3).unwrap();

    #[test]
    fn ensure_valid_flat_len() {
        assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(NZ_2, NZ_3, NZ_U8_2, 12),
            Ok(())
        );
        assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(NZ_2, NZ_3, NZ_U8_2, 11),
            Err(InvalidLenError::BufferMismatch {
                expected: 12,
                actual: 11
            })
        );
        // The multiplication with `pixel_elements` overflows even `u64` (on any target)
        assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(
                NonZeroU32::MAX,
                NonZeroU32::MAX,
                NonZeroU8::MAX,
                usize::MAX
            ),
            Err(InvalidLenError::Overflow(OverflowBufferLenError::new(
                NonZeroU32::MAX,
                NonZeroU32::MAX,
                NonZeroU8::MAX,
                NonZeroU8::MIN,
            )))
        );
    }

    #[test]
    fn ensure_valid_len() -> TestResult {
        assert_eq!(
            DynamicImageChannelView::ensure_valid_len(NZ_2, NZ_3, 6),
            Ok(())
        );
        assert_eq!(
            DynamicImageChannelView::ensure_valid_len(NZ_2, NZ_3, 18),
            Err(InvalidLenError::BufferMismatch {
                expected: 6,
                actual: 18
            })
        );
        // `width * height` fits `u64`, but can overflow a smaller `usize` (on 64-bit targets,
        // where it fits, the buffer length mismatches instead)
        let expected = if cfg!(target_pointer_width = "64") {
            InvalidLenError::BufferMismatch {
                expected: usize::try_from(u64::from(u32::MAX) * u64::from(u32::MAX))?,
                actual: usize::MAX,
            }
        } else {
            InvalidLenError::Overflow(OverflowBufferLenError::new(
                NonZeroU32::MAX,
                NonZeroU32::MAX,
                NonZeroU8::MIN,
                NonZeroU8::MIN,
            ))
        };
        assert_eq!(
            DynamicImageChannelView::ensure_valid_len(NonZeroU32::MAX, NonZeroU32::MAX, usize::MAX),
            Err(expected)
        );
        Ok(())
    }

    #[test]
    fn ensure_single_channel() -> TestResult {
        let single = DynamicImage::from(Image::<[u8; 3], 1>::new_vec(vec![[1, 2, 3]], NZ_1, NZ_1)?);
        assert_eq!(single.ensure_single_channel(), Ok(()));

        let planar = DynamicImage::from(Image::<u8, 3>::new_vec(vec![1, 2, 3], NZ_1, NZ_1)?);
        assert_eq!(
            planar.ensure_single_channel(),
            Err(IncompatibleChannelCount {
                expected: NonZeroU8::MIN,
                actual: NonZeroUsize::new(3).ok_or("zero")?,
            })
        );
        Ok(())
    }

    #[test]
    fn image_with_other_dimensions_is_not_uniform() -> TestResult {
        let ch1 = ImageChannel::<u8>::new_vec(vec![1, 1], NZ_1, NZ_2)?;
        let ch2 = ImageChannel::<u8>::new_vec(vec![1], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from_channels(ch1.into(), [ch2.into()]);
        assert!(!dynamic.is_uniform());
        Ok(())
    }

    #[test]
    fn image_with_other_bytes_per_primitive_is_not_uniform() -> TestResult {
        let ch1 = ImageChannel::<u8>::new_vec(vec![1], NZ_1, NZ_1)?;
        let ch2 = ImageChannel::<u16>::new_vec(vec![1], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from_channels(ch1.into(), [ch2.into()]);
        assert!(!dynamic.is_uniform());
        Ok(())
    }

    #[test]
    fn image_as_dynamic_is_uniform() -> TestResult {
        let image = Image::<u8, 3>::new_vec(vec![0, 1, 2, 3, 4, 5], NZ_1, NZ_2)?;
        let dynamic = DynamicImage::from(image);
        assert!(dynamic.is_uniform());
        Ok(())
    }

    #[test]
    fn first_mapped_returns_ref() -> TestResult {
        let ch = Image::<u8, 3>::new_vec(vec![1, 2, 3], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(ch);
        let (first, mut rest) = dynamic.mapped_channels(DynamicImageChannel::buffer_flat_bytes);
        assert_eq!(1, first[0]);
        assert_eq!(2, rest.next().ok_or("missing channel")?[0]);
        assert_eq!(3, rest.next().ok_or("missing channel")?[0]);
        Ok(())
    }

    #[test]
    fn first_mapped_returns_mut() -> TestResult {
        let ch = Image::<u8, 3>::new_vec(vec![1, 2, 3], NZ_1, NZ_1)?;
        let mut dynamic = DynamicImage::from(ch);
        let (first, mut rest) = dynamic.mapped_channels_mut(|x| x.make_mut_flat::<u8>());
        first?[0] = 4;
        assert_eq!(2, rest.next().ok_or("missing channel")??[0]);
        assert_eq!(3, rest.next().ok_or("missing channel")??[0]);
        assert_eq!(dynamic.first().buffer_flat::<u8>(), Ok(&[4][..]));
        Ok(())
    }

    #[test]
    fn buffer_flat_checks_bytes_per_primitive() -> TestResult {
        let channel = DynamicImageChannel::new_vec(vec![[1u16, 2]], NZ_1, NZ_1)?;
        assert_eq!(channel.bytes_per_primitive().get(), 2);
        assert_eq!(channel.len_flat(), 2);
        assert_eq!(channel.buffer_flat::<u16>(), Ok(&[1, 2][..]));
        assert_matches!(channel.buffer_flat::<u8>(), Err(_));
        assert_matches!(channel.buffer_flat::<f32>(), Err(_));
        assert_eq!(channel.buffer_flat_bytes(), &[1, 0, 2, 0]);
        Ok(())
    }

    #[test]
    fn make_mut_flat_copies_shared_buffer() -> TestResult {
        let data = std::sync::Arc::<[u8]>::from([1].as_slice());
        let image = Image::<u8, 1>::new_arc(data, NZ_1, NZ_1)?;
        let data_ptr = image.buffer().as_ptr();
        let mut dynamic = DynamicImage::from(image);
        let clone = dynamic.clone();

        assert_matches!(dynamic[0].make_mut_flat::<u16>(), Err(_));
        let buffer = dynamic[0].make_mut_flat::<u8>()?;
        buffer[0] = 2;
        assert_ne!(buffer.as_ptr(), data_ptr);
        assert_eq!(clone.first().buffer_flat::<u8>(), Ok(&[1][..]));
        Ok(())
    }

    #[test]
    fn borrow_from_mut_dynamic_image() -> TestResult {
        let pixel_data = vec![1];
        let pixel_data_ptr = pixel_data.as_ptr() as usize;
        let mut image = DynamicImage::from(Image::<u8, 1>::new_vec(pixel_data, NZ_1, NZ_1)?);
        let mut image_back: ImageView<'_, u8, 1> = (&mut image).try_into()?;

        assert_eq!(pixel_data_ptr, image_back.make_mut()[0].as_ptr() as usize);
        image_back.make_mut()[0][0] = 2;
        drop(image_back);
        assert_eq!(image.first().buffer_flat::<u8>(), Ok(&[2][..]));
        Ok(())
    }

    #[test]
    fn borrow_from_mut_dynamic_image_copies_shared_buffer_into_dynamic_image() -> TestResult {
        let data = std::sync::Arc::<[u8]>::from([1].as_slice());
        let data_ptr = data.as_ptr();
        let mut dynamic = DynamicImage::from(Image::<u8, 1>::new_arc(data, NZ_1, NZ_1)?);
        let clone = dynamic.clone();

        let mut view = ImageView::<u8, 1>::try_from(&mut dynamic)?;
        assert_eq!(view.buffer().as_ptr(), data_ptr, "Reading doesn't copy");
        view.make_mut_flat()[0] = 2;
        assert_ne!(view.buffer().as_ptr(), data_ptr);

        let view_clone = view.clone();
        view.make_mut_flat()[0] = 3;
        assert_eq!(view_clone.buffer(), &[2], "Clones of mutable views copy");
        drop((view, view_clone));

        assert_eq!(dynamic.first().buffer_flat::<u8>(), Ok(&[3][..]));
        assert_eq!(clone.first().buffer_flat::<u8>(), Ok(&[1][..]));
        Ok(())
    }

    #[test]
    fn borrow_from_dynamic_image_copies_on_write() -> TestResult {
        let dynamic = DynamicImage::from(Image::<u8, 2>::new_vec(vec![1, 2], NZ_1, NZ_1)?);
        let mut view = ImageView::<u8, 2>::try_from(&dynamic)?;
        assert_eq!(
            view.buffers()[1].as_ptr(),
            dynamic[1].buffer_flat::<u8>()?.as_ptr()
        );
        let [_, second] = view.make_mut();
        second[0] = 3;
        assert_eq!(view.buffers(), [[1], [3]]);
        assert_eq!(dynamic[1].buffer_flat::<u8>(), Ok(&[2][..]));
        Ok(())
    }

    #[test]
    fn borrowed_image_into_dynamic_image_and_back() -> TestResult {
        let mut data = [1u16, 2];
        let data_ptr = data.as_ptr();
        let view = ImageView::<u16, 2>::new_slice_mut(&mut data, NZ_1, NZ_1)?;
        let mut dynamic = DynamicImageView::from(view);
        dynamic[1].make_mut_flat::<u16>()?[0] = 3;
        let back = ImageView::<u16, 2>::try_from(dynamic)?;
        assert_eq!(back.buffers()[0].as_ptr(), data_ptr);
        drop(back);
        assert_eq!(data, [1, 3]);
        Ok(())
    }

    #[test]
    fn borrow_from_different_size_mut_dynamic_image() -> TestResult {
        let mut dynamic = DynamicImage::from(Image::<u8, 2>::new_vec(vec![0, 1], NZ_1, NZ_1)?);
        dynamic[1] = ImageChannel::<u8>::new_vec(vec![0, 1], NZ_1, NZ_2)?.into();

        assert_matches!(
            ImageView::<u8, 2>::try_from(&mut dynamic),
            Err(IncompatibleImageError::MixedImageSizes { .. })
        );
        assert_matches!(
            ImageView::<u8, 2>::try_from(&dynamic),
            Err(IncompatibleImageError::MixedImageSizes { .. })
        );
        assert_matches!(
            dynamic.try_cast::<u8, 2>(),
            Err(IncompatibleImageError::MixedImageSizes { .. })
        );
        assert_matches!(
            dynamic.try_cast_mut::<u8, 2>(),
            Err(IncompatibleImageError::MixedImageSizes { .. })
        );
        Ok(())
    }

    #[test]
    fn try_cast_dynamic_image() -> TestResult {
        let data = vec![1u8, 2];
        let data_ptr = data.as_ptr() as usize;
        let dynamic = DynamicImage::from(Image::<u8, 2>::new_vec(data, NZ_1, NZ_1)?);
        let image = dynamic.try_cast::<u8, 2>()?;
        assert_eq!(image.buffers(), [[1], [2]]);
        assert_eq!(
            image.buffers()[0].as_ptr() as usize,
            data_ptr,
            "Should share the buffer"
        );

        assert_matches!(
            dynamic.try_cast::<u8, 1>(),
            Err(IncompatibleImageError::ChannelCount(_))
        );
        assert_matches!(
            dynamic.try_cast::<u16, 2>(),
            Err(IncompatibleImageError::Channel(_))
        );
        assert_matches!(
            dynamic.try_cast::<[u8; 2], 2>(),
            Err(IncompatibleImageError::Channel(_))
        );
        Ok(())
    }

    #[test]
    fn try_cast_mut_dynamic_image_writes_into_dynamic_image() -> TestResult {
        let mut dynamic = DynamicImage::from(Image::<u16, 2>::new_vec(vec![1, 2], NZ_1, NZ_1)?);
        let clone = dynamic.clone();
        assert_matches!(
            dynamic.try_cast_mut::<u8, 2>(),
            Err(IncompatibleImageError::Channel(_))
        );

        let image = dynamic.try_cast_mut::<u16, 2>()?;
        image.make_mut()[1][0] = 3;
        assert_eq!(dynamic[1].buffer_flat::<u16>(), Ok(&[3][..]));
        assert_eq!(clone[1].buffer_flat::<u16>(), Ok(&[2][..]));
        Ok(())
    }

    #[test]
    fn try_into_typed_dynamic_image() -> TestResult {
        let rgb = Image::<[u8; 3], 1>::new_vec(vec![[1, 2, 3]], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(rgb);
        assert_matches!(
            dynamic.clone().try_into_typed::<u8, 1>(),
            Err(IncompatibleImageError::Channel(_))
        );
        let rgb_back = dynamic.try_into_typed::<[u8; 3], 1>()?;
        assert_eq!(rgb_back.into_vec(), vec![[1, 2, 3]]);
        Ok(())
    }

    #[test]
    fn create_from_luma_u8() -> TestResult {
        let luma = Image::<u8, 1>::new_vec(vec![1], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(luma);
        assert_eq!(1, dynamic.channels.len());
        let luma_back: Image<u8, 1> = dynamic.try_into()?;
        assert_eq!(luma_back.into_vec(), vec![1]);
        Ok(())
    }

    #[test]
    fn create_from_luma_rgb16_interleaved() -> TestResult {
        let luma = Image::<[u16; 3], 1>::new_vec(vec![[1u16, 2, 3]], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(luma);
        assert_eq!(1, dynamic.channels.len());
        let luma_back: Image<[u16; 3], 1> = dynamic.try_into()?;
        assert_eq!(luma_back.into_vec(), vec![[1u16, 2, 3]]);
        Ok(())
    }

    #[test]
    fn create_from_rgb8_interleaved() -> TestResult {
        let rgb = Image::<[u8; 3], 1>::new_vec(vec![[1u8, 2, 3]], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(rgb);
        assert_eq!(1, dynamic.channels.len());
        assert_eq!(1, dynamic[0].bytes_per_primitive().get());
        assert_eq!(3, dynamic[0].pixel_elements().get());
        assert_eq!(Ok([[1u8, 2, 3]].as_slice()), dynamic[0].buffer::<[u8; 3]>());
        let rgb_back: Image<[u8; 3], 1> = dynamic.try_into()?;
        assert_eq!(rgb_back.into_vec(), vec![[1u8, 2, 3]]);
        Ok(())
    }
    #[test]
    fn create_from_luma_rgb8_planar() -> TestResult {
        let luma = Image::<u8, 3>::new_vec(vec![1u8, 2, 3], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(luma);
        assert_eq!(3, dynamic.channels.len());
        let luma_back: Image<u8, 3> = dynamic.try_into()?;
        assert_eq!(luma_back.into_vec(), vec![1u8, 2, 3]);
        Ok(())
    }

    #[test]
    fn clone_dynamic_image() -> TestResult {
        let luma = Image::<u8, 1>::new_vec(vec![1, 2, 3, 4], NZ_2, NZ_2)?;
        let dynamic = DynamicImage::from(luma);
        let cloned = dynamic.clone();

        // Verify both can be converted back to the same image
        let luma_back: Image<u8, 1> = dynamic.try_into()?;
        {
            let ref_luma: ImageView<u8, 1> = (&cloned).try_into()?;
            assert_eq!(ref_luma.dimensions(), (NZ_2, NZ_2));
        }
        let luma_cloned: Image<u8, 1> = cloned.try_into()?;
        let vec_back = luma_back.into_vec();
        let vec_cloned = luma_cloned.into_vec();
        assert_eq!(vec_back, vec_cloned);
        assert_eq!(vec_cloned, vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn create_from_incompatible_image() -> TestResult {
        let luma = Image::<u8, 1>::new_vec(vec![42], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(luma.clone());
        assert_matches!(
            Image::<u16, 1>::try_from(dynamic.clone()),
            Err(IncompatibleImageError::Channel(_))
        );
        assert_matches!(
            Result::<Image<u16, 1>, IncompatibleImage<_>>::from(dynamic),
            Err(IncompatibleImage {
                image,
                error:
                    IncompatibleImageError::Channel(IncompatibleChannelError {
                        actual: PixelLayout {
                            elements: NonZeroU8::MIN,
                            bytes_per_primitive: NonZeroUsize::MIN
                        },
                        required: Some(PixelLayout {
                            elements: NonZeroU8::MIN,
                            bytes_per_primitive: second_bytes_per_primitive
                        }),
                        target_pixel_type: "u16",
                        ..
                    }),
                ..
            }) if image == DynamicImage::from(luma) && second_bytes_per_primitive.get() == 2
        );
        Ok(())
    }

    #[test]
    fn create_from_image_with_other_pixel_elements() -> TestResult {
        let rgb = Image::<[u8; 3], 1>::new_vec(vec![[1u8, 2, 3]], NZ_1, NZ_1)?;
        let dynamic = DynamicImage::from(rgb);
        assert_matches!(
            ImageView::<u8, 1>::try_from(&dynamic),
            Err(IncompatibleImageError::Channel(IncompatibleChannelError {
                actual: PixelLayout {
                    elements: NZ_U8_3,
                    bytes_per_primitive: first_bytes_per_primitive
                },
                required: Some(PixelLayout {
                    elements: NonZeroU8::MIN,
                    bytes_per_primitive: NonZeroUsize::MIN
                }),
                aligned: true,
                target_pixel_type: "u8",
            })) if first_bytes_per_primitive.get() == 1
        );
        Ok(())
    }
}
