use std::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    slice,
};

use bytemuck::Pod;

use crate::{
    ImageChannelView, InvalidLenError, PixelType, PixelTypePrimitive,
    error::{IncompatibleChannelError, OverflowBufferLenError, PixelLayout},
    vtable::ImageChannelVTable,
};

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
/// impl PixelTypePrimitive for NonZero {
///     const TYPE_HINT: [u8; 8] = *b"nonzero\0";
/// }
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
    /// Pointer to the buffer, which is `len_flat()` primitives of `primitive_size()` bytes long
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
    /// Creates a channel from its raw parts, e.g. to support a custom storage. `vtable` is created
    /// for the primitive `T` of the buffer by
    /// [`ImageChannelVTable::new::<T>`](crate::ImageChannelVTable::new).
    ///
    /// # Safety
    /// The following invariants have to hold:
    /// - The size of the buffer in bytes (`width * height * pixel_elements * size_of::<T>()`)
    ///   doesn't exceed `isize::MAX` (like any slice), and the multiplications to calculate it
    ///   don't overflow, which implies that `width * height * pixel_elements` matches the buffer
    ///   length. [`DynamicImageChannelView::ensure_valid_flat_len`] checks the length, e.g. for
    ///   untrusted input.
    /// - `ptr` is aligned for `T`.
    /// - `ptr` points to an initialized buffer of that size, which stays valid until the `drop`
    ///   function of the vtable is called and at least for `'a`. The buffer can be shared between
    ///   instances.
    /// - Nothing else writes the buffer, while the channel (or a clone sharing the buffer) exists.
    ///   The channel itself only writes it after calling the `make_mut` function of the vtable,
    ///   which makes it exclusive: Until the channel is cloned or dropped, nothing else accesses
    ///   the buffer.
    /// - The functions of the vtable are able to work with `ptr` and `data` and fulfill the
    ///   contract described in [`ImageChannelVTable::new`](crate::ImageChannelVTable::new).
    ///   Any bytes written through a `Pod` view of the buffer have to be valid for the vtable
    ///   (e.g. `T` is `Pod` itself).
    /// - `ptr` and `data` can be sent to and shared between threads. The `clone` function of the
    ///   vtable can be called concurrently, as channels are `Sync`.
    ///
    /// # Panics
    /// In debug builds, panics if `width * height * pixel_elements` overflows `usize` or the
    /// buffer size in bytes exceeds `isize::MAX`.
    #[must_use]
    pub unsafe fn from_raw_parts(
        ptr: *const u8,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        vtable: &'static ImageChannelVTable,
        data: *mut (),
    ) -> Self {
        // Not dropped, if a check panics, as the caller still owns the raw parts
        let channel = ManuallyDrop::new(Self {
            ptr,
            width,
            height,
            vtable,
            data,
            pixel_elements,
            lifetime: PhantomData,
        });
        debug_assert_eq!(
            Ok(()),
            Self::ensure_valid_byte_len(width, height, pixel_elements, channel.primitive_size()),
            "The buffer size of the raw parts exceeds isize::MAX"
        );
        debug_assert!(
            channel.primitive_alignment().is_power_of_two()
                && ptr.addr() % channel.primitive_alignment().get() == 0,
            "The pointer of the raw parts isn't aligned for the primitive of the vtable"
        );
        ManuallyDrop::into_inner(channel)
    }

    /// Const version of [`DynamicImageChannelView::from_raw_parts`] without its debug checks of
    /// the byte length and alignment (they call the vtable, which isn't possible in const).
    /// The `debug_assert!`s just help catch developer mistakes, so they are not absolutely
    /// necessary.
    ///
    /// # Safety
    /// See [`DynamicImageChannelView::from_raw_parts`]
    pub(crate) const unsafe fn from_raw_parts_const(
        ptr: *const u8,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        vtable: &'static ImageChannelVTable,
        data: *mut (),
    ) -> Self {
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

    /// Read access to the buffer pointer, e.g. to support a custom storage or implement the
    /// functions of an [`ImageChannelVTable`](crate::ImageChannelVTable).
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
    pub fn primitive_size(&self) -> NonZeroUsize {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.vtable.primitive_size)(self) }
    }

    /// Alignment in bytes, which every buffer of the channel is aligned to
    #[must_use]
    pub fn primitive_alignment(&self) -> NonZeroUsize {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.vtable.primitive_alignment)(self) }
    }

    /// Name of the primitive the vtable was created for
    /// ([`PixelTypePrimitive::TYPE_HINT`](crate::PixelTypePrimitive::TYPE_HINT)), e.g. `"u8"`.
    ///
    /// It's only a hint: The buffer can be viewed as any [`Pod`] primitive with the same layout,
    /// and channels created by `clone` or `make_mut` may report the hint of another primitive
    /// with the same layout.
    #[must_use]
    pub const fn type_hint(&self) -> &'static str {
        let vtable: &'static ImageChannelVTable = self.vtable;
        match crate::pixel::parse_type_hint(&vtable.type_hint) {
            Some(name) => name,
            None => unreachable!(),
        }
    }

    /// Debug check of the constraints of [`ImageChannelVTable`](crate::ImageChannelVTable) for
    /// `self`, which was created by `operation` (`clone` or `make_mut`) of a channel with
    /// `layout`
    #[cfg(debug_assertions)]
    fn assert_derived_layout(
        &self,
        (layout, dims): (PixelLayout, (NonZeroU32, NonZeroU32)),
        operation: &str,
    ) {
        assert!(
            self.dimensions() == dims,
            "vtable.{operation} changed the dimensions of the channel"
        );

        let derived = self.pixel_layout();
        assert!(
            derived.fits(layout),
            "vtable.{operation} changed the layout from {layout} to {derived}"
        );
        assert!(
            self.ptr.addr() % derived.primitive_alignment.get() == 0,
            "vtable.{operation} returned a buffer, which isn't aligned to primitive_alignment"
        );
    }

    /// `pixel_elements`, `primitive_size` and `primitive_alignment` of the channel
    #[must_use]
    pub fn pixel_layout(&self) -> PixelLayout {
        PixelLayout::new(
            self.pixel_elements,
            self.primitive_size(),
            self.primitive_alignment(),
        )
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
        // make_mut might reallocate, but keeps the primitive (contract of the vtable). Size and
        // alignment are checked again in debug builds.
        // Safety: The buffer is exclusive. The primitive is forwarded to the caller
        unsafe { self.buffer_flat_mut_unchecked() }
    }

    /// Mutable access to the buffer without copying it. `None`, if writing requires a copy,
    /// because the buffer is shared (e.g. with a clone) or borrowed immutably, or the storage
    /// can't tell (see `try_make_mut` of [`ImageChannelVTable::new`]).
    /// [`DynamicImageChannelView::make_mut_flat`] copies in that case.
    ///
    /// Fails, if `pixel_elements`, the size or the alignment of `T::Primitive` don't match
    pub fn buffer_mut<T: PixelType + Pod>(
        &mut self,
    ) -> Result<Option<&mut [T]>, IncompatibleChannelError> {
        self.ensure_fits_pixel::<T>()?;
        Ok(if self.try_make_exclusive() {
            // Safety: The buffer is exclusive and the layout is checked. Pod accepts any bit
            // pattern
            Some(unsafe { self.buffer_mut_unchecked() })
        } else {
            None
        })
    }

    /// Flat version of [`DynamicImageChannelView::buffer_mut`]
    ///
    /// Fails, if the size or the alignment of `T` don't match the primitive of the channel
    pub fn buffer_flat_mut<T: PixelTypePrimitive + Pod>(
        &mut self,
    ) -> Result<Option<&mut [T]>, IncompatibleChannelError> {
        self.ensure_fits_primitive::<T>()?;
        Ok(if self.try_make_exclusive() {
            // Safety: The buffer is exclusive, size and alignment are checked. Pod accepts any
            // bit pattern
            Some(unsafe { self.buffer_flat_mut_unchecked() })
        } else {
            None
        })
    }

    /// # Safety
    /// The buffer has to be exclusive (see [`DynamicImageChannelView::make_exclusive`] and
    /// [`DynamicImageChannelView::try_make_exclusive`]) and `T` has to fit like for
    /// [`DynamicImageChannelView::buffer_unchecked`]
    pub(crate) unsafe fn buffer_mut_unchecked<T: PixelType>(&mut self) -> &mut [T] {
        debug_assert_eq!(Ok(()), self.ensure_fits_pixel::<T>());
        // Safety: Forwarded to the caller. The length is valid due to the invariants of the channel
        unsafe { slice::from_raw_parts_mut(self.ptr.cast::<T>().cast_mut(), self.len()) }
    }

    /// # Safety
    /// The buffer has to be exclusive (see [`DynamicImageChannelView::buffer_mut_unchecked`]) and
    /// `T` has to fit like for [`DynamicImageChannelView::buffer_flat_unchecked`]
    pub(crate) unsafe fn buffer_flat_mut_unchecked<T: PixelTypePrimitive>(&mut self) -> &mut [T] {
        debug_assert_eq!(Ok(()), self.ensure_fits_primitive::<T>());
        // Safety: Forwarded to the caller. The length is valid due to the invariants of the channel
        unsafe { slice::from_raw_parts_mut(self.ptr.cast::<T>().cast_mut(), self.len_flat()) }
    }

    /// Calls `vtable.make_mut`, so the buffer is exclusively owned by the channel afterwards
    pub(crate) fn make_exclusive(&mut self) {
        #[cfg(debug_assertions)]
        let before = (self.pixel_layout(), self.dimensions());
        // Safety: The vtable is able to work with the channel
        unsafe { (self.vtable.make_mut)(self) };
        #[cfg(debug_assertions)]
        self.assert_derived_layout(before, "make_mut");
    }

    /// Calls `vtable.try_make_mut`: Whether the buffer is exclusively owned by the channel
    /// afterwards, without copying it
    pub(crate) fn try_make_exclusive(&mut self) -> bool {
        #[cfg(debug_assertions)]
        let before = (self.pixel_layout(), self.dimensions());
        // Safety: The vtable is able to work with the channel
        let exclusive = unsafe { (self.vtable.try_make_mut)(self) };
        #[cfg(debug_assertions)]
        self.assert_derived_layout(before, "try_make_mut");
        exclusive
    }

    #[must_use]
    pub fn buffer_flat_bytes(&self) -> &[u8] {
        let len = self.len_flat() * self.primitive_size().get();
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
        self.ensure_fits_layout::<TP>(PixelLayout::of::<TP>())
    }

    /// Fails with the reason, if the flat buffer can't be viewed as `T`. In contrast to
    /// [`DynamicImageChannelView::ensure_fits_pixel`], `pixel_elements` doesn't have to match.
    pub(crate) fn ensure_fits_primitive<T: PixelTypePrimitive>(
        &self,
    ) -> Result<(), IncompatibleChannelError> {
        self.ensure_fits_layout::<T>(PixelLayout::of_primitive::<T>(self.pixel_elements()))
    }

    /// Fails with the reason, if the channel doesn't fit the `required` layout of `TTarget`.
    /// The alignment of the channel is guaranteed for all of its buffers, so typed views stay
    /// aligned, when `clone` or `make_mut` of the vtable reallocate.
    fn ensure_fits_layout<TTarget>(
        &self,
        required: PixelLayout,
    ) -> Result<(), IncompatibleChannelError> {
        let actual = self.pixel_layout();
        if actual.fits(required) {
            Ok(())
        } else {
            Err(IncompatibleChannelError {
                actual,
                required: Some(required),
                target_pixel_type: std::any::type_name::<TTarget>(),
            })
        }
    }

    /// Checks that the size of the buffer in bytes (`width * height * pixel_elements *
    /// primitive_size`) doesn't exceed `isize::MAX`, which [`DynamicImageChannelView::from_raw_parts`]
    /// requires
    fn ensure_valid_byte_len(
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        primitive_size: NonZeroUsize,
    ) -> Result<(), OverflowBufferLenError> {
        let len = Self::flat_len_checked(width, height, pixel_elements, NonZeroU8::MIN)?;
        len.checked_mul(primitive_size.get())
            .filter(|&bytes| bytes <= isize::MAX.unsigned_abs())
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
        #[cfg(debug_assertions)]
        let before = (self.pixel_layout(), self.dimensions());
        // Safety: The vtable is able to work with the channel
        let clone = unsafe { (self.vtable.clone)(self) };
        #[cfg(debug_assertions)]
        clone.assert_derived_layout(before, "clone");
        clone
    }
}

/// Channels are equal, if their dimensions, `pixel_elements`, `primitive_size` and bytes are equal
impl PartialEq for DynamicImageChannelView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions() == other.dimensions()
            && self.pixel_elements() == other.pixel_elements()
            && self.primitive_size() == other.primitive_size()
            && self.buffer_flat_bytes() == other.buffer_flat_bytes()
    }
}

impl Debug for DynamicImageChannelView<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicImageChannel")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("primitive_size", &self.primitive_size())
            .field("type_hint", &self.type_hint())
            .field("pixel_elements", &self.pixel_elements())
            .finish()
    }
}

// Safety: The primitives are Send + Sync and storages have to be thread safe (invariant of `from_raw_parts`)
unsafe impl Send for DynamicImageChannelView<'_> {}
unsafe impl Sync for DynamicImageChannelView<'_> {}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::num::{NonZeroU8, NonZeroU32};

    use testresult::TestResult;

    use super::*;
    use crate::{DynamicImage, Image};

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    const NZ_3: NonZeroU32 = NonZeroU32::new(3).unwrap();
    const NZ_U8_2: NonZeroU8 = NonZeroU8::new(2).unwrap();

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
    fn type_hint_of_builtin_primitives_is_their_name() -> TestResult {
        fn assert_type_hint<T: crate::PixelTypePrimitive + Pod>() -> TestResult {
            let channel = DynamicImageChannel::new_vec(vec![T::zeroed()], NZ_1, NZ_1)?;
            assert_eq!(channel.type_hint(), std::any::type_name::<T>());
            Ok(())
        }
        assert_type_hint::<u8>()?;
        assert_type_hint::<i8>()?;
        assert_type_hint::<u16>()?;
        assert_type_hint::<i16>()?;
        assert_type_hint::<u32>()?;
        assert_type_hint::<i32>()?;
        assert_type_hint::<u64>()?;
        assert_type_hint::<i64>()?;
        assert_type_hint::<usize>()?;
        assert_type_hint::<isize>()?;
        assert_type_hint::<f32>()?;
        assert_type_hint::<f64>()?;
        Ok(())
    }

    #[test]
    fn type_hint_is_parsed_until_the_first_nul() {
        use crate::pixel::parse_type_hint;
        assert_eq!(parse_type_hint(b"u8\0\0\0\0\0\0"), Some("u8"));
        assert_eq!(parse_type_hint(b"RGBA8888"), Some("RGBA8888"));
        assert_eq!(parse_type_hint(b"\0\0\0\0\0\0\0\0"), Some(""));
        // Bytes after the first nul byte are ignored
        assert_eq!(parse_type_hint(b"ab\0\xFF\xFF\xFF\xFF\xFF"), Some("ab"));
        // Multi byte characters, also if cut off by the end
        assert_eq!(
            parse_type_hint("µ8\0\0\0\0\0".as_bytes().try_into().unwrap()),
            Some("µ8")
        );
        assert_eq!(parse_type_hint(b"1234567\xC2"), None);
        assert_eq!(parse_type_hint(b"\xFF\0\0\0\0\0\0\0"), None);
    }

    #[test]
    fn buffer_flat_checks_primitive_size() -> TestResult {
        let channel = DynamicImageChannel::new_vec(vec![[1u16, 2]], NZ_1, NZ_1)?;
        assert_eq!(channel.primitive_size().get(), 2);
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
}
