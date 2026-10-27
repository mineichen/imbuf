use std::{
    fmt::{self, Debug, Formatter},
    iter::{self, once},
    marker::PhantomData,
    mem::MaybeUninit,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    ops::Deref,
    ptr, slice, vec,
};

use bytemuck::Pod;

use crate::{
    ImageChannelView, ImageView, PixelType, PixelTypePrimitive,
    channel::{ImageChannelVTable, calc_pixel_len_flat, calc_pixel_len_packed},
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

    /// Checks, if all channels have the same width, height, `pixel_elements` and `primitive_size`
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        let (first, mut rest) =
            self.mapped_channels(|x| (x.primitive_size(), x.pixel_elements(), x.dimensions()));
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
/// [`DynamicImageChannelView::buffer_flat`], [`DynamicImageChannelView::try_cast`] or
/// `ImageChannelView::try_from`. For the same reason, only [`ImageChannelView`]s of `Pod`
/// primitives can be converted into a `DynamicImageChannelView`:
///
/// ```compile_fail,E0277
/// use std::num::{NonZeroU8, NonZeroU32};
/// use imbuf::{DynamicImageChannel, ImageChannel, PixelTypePrimitive};
///
/// #[derive(Clone, Copy, PartialEq)]
/// struct NonZero(NonZeroU8);
/// impl PixelTypePrimitive for NonZero {}
///
/// let channel = ImageChannel::new_vec(vec![NonZero(NonZeroU8::MIN)], NonZeroU32::MIN, NonZeroU32::MIN);
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
    /// `vtable.primitive_size()` long
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

impl DynamicImageChannel {
    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height and `pixel_elements`.
    #[must_use]
    pub fn new_vec<T: PixelTypePrimitive + Pod>(
        data: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Self {
        Self::from_vec(data, width, height, pixel_elements)
    }
}

impl<'a> DynamicImageChannelView<'a> {
    /// Creates a channel from its raw parts, e.g. to support a custom storage.
    ///
    /// # Safety
    /// The following invariants have to hold:
    /// - The size of the buffer in bytes (`width * height * pixel_elements * vtable.primitive_size()`)
    ///   doesn't exceed `isize::MAX`.
    /// - `ptr` points to an initialized buffer of that size, which stays valid until `vtable.drop`
    ///   is called and at least for `'a`. The buffer can be shared between instances.
    /// - `vtable` is able to work with `ptr` and `data` (see [`ImageChannelVTable`](crate::ImageChannelVTable)).
    ///   Any bytes written through a `Pod` view of the buffer have to be valid for the vtable
    ///   (e.g. the primitive of the vtable is `Pod` itself).
    /// - `ptr` and `data` can be sent to and shared between threads.
    #[must_use]
    pub unsafe fn from_raw_parts(
        ptr: *const u8,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        vtable: &'static ImageChannelVTable,
        data: *mut (),
    ) -> Self {
        #[cfg(debug_assertions)]
        let _asserted = usize::try_from(
            u64::from(height.get())
                .strict_mul(u64::from(width.get()).strict_mul(u64::from(pixel_elements.get()))),
        )
        .expect("Expected valid raw parts")
        .strict_mul((vtable.primitive_size)().get());
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
    pub fn primitive_size(&self) -> NonZeroUsize {
        (self.vtable.primitive_size)()
    }

    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        calc_pixel_len_packed(self.width, self.height)
    }

    /// Number of primitives in the buffer (`width * height * pixel_elements`)
    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub const fn len_flat(&self) -> usize {
        calc_pixel_len_flat(self.width, self.height, self.pixel_elements)
    }

    /// Returns `None`, if the size or the alignment of `T` don't match the primitive of the channel
    #[must_use]
    pub fn buffer_flat<T: PixelTypePrimitive + Pod>(&self) -> Option<&[T]> {
        // Safety: Size and alignment are checked. Pod accepts any bit pattern
        self.fits_primitive::<T>()
            .then(|| unsafe { self.buffer_flat_unchecked() })
    }

    /// # Safety
    /// `T` must have the size of the channels primitive, `ptr` must be aligned for `T` and the
    /// buffer must contain valid values of `T`.
    #[must_use]
    pub unsafe fn buffer_flat_unchecked<T>(&self) -> &[T] {
        debug_assert!(
            self.fits_primitive::<T>(),
            "The size or alignment of {} doesn't match the channel",
            std::any::type_name::<T>()
        );
        // Safety: Forwarded to the caller. The length is valid due to the invariants of the channel
        unsafe { slice::from_raw_parts(self.ptr.cast::<T>(), self.len_flat()) }
    }

    /// Mutable access to the buffer, which is copied if it is shared (copy on write).
    ///
    /// Returns `None`, if the size or the alignment of `T` don't match the primitive of the channel
    pub fn make_mut_flat<T: PixelTypePrimitive + Pod>(&mut self) -> Option<&mut [T]> {
        if !self.fits_primitive::<T>() {
            return None;
        }
        Some(unsafe { self.make_mut_flat_unchecked() })
    }

    pub(crate) unsafe fn make_mut_flat_unchecked<T: PixelTypePrimitive>(&mut self) -> &mut [T] {
        self.make_exclusive();
        // Safety: Size and alignment are checked after make_mut, which might reallocate
        debug_assert!(self.fits_primitive::<T>());
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
        let len = self.len_flat() * self.primitive_size().get();
        // Safety: The buffer is initialized (invariant of the channel)
        unsafe { slice::from_raw_parts(self.ptr, len) }
    }

    /// Try to view this channel as a `ImageChannelView<TP>`.
    ///
    /// Returns `None`, if `pixel_elements`, the size or the alignment of `TP::Primitive` don't match.
    /// Use [`ImageChannelView::view`] on the result to get a borrowed channel by value.
    #[must_use]
    pub fn try_cast<TP: PixelType>(&self) -> Option<&ImageChannelView<'a, TP>>
    where
        TP::Primitive: Pod,
    {
        // Safety: `ImageChannelView` is repr(transparent) over `DynamicImageChannelView`.
        // The layout is checked and Pod accepts any bit pattern
        self.fits_pixel::<TP>()
            .then(|| unsafe { &*ptr::from_ref(self).cast::<ImageChannelView<'a, TP>>() })
    }

    /// Mutable version of [`DynamicImageChannelView::try_cast`]
    #[must_use]
    pub fn try_cast_mut<TP: PixelType>(&mut self) -> Option<&mut ImageChannelView<'a, TP>>
    where
        TP::Primitive: Pod,
    {
        // Safety: See try_cast
        self.fits_pixel::<TP>()
            .then(|| unsafe { &mut *ptr::from_mut(self).cast::<ImageChannelView<'a, TP>>() })
    }

    /// Checks, if the buffer can be viewed as `T`
    pub(crate) fn fits_primitive<T>(&self) -> bool {
        self.primitive_size().get() == size_of::<T>() && self.ptr.cast::<T>().is_aligned()
    }

    /// Checks, if the buffer can be viewed as `TP`
    pub(crate) fn fits_pixel<TP: PixelType>(&self) -> bool {
        self.pixel_elements() == TP::ELEMENTS && self.fits_primitive::<TP::Primitive>()
    }
}

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

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
#[error("IncompatibleImageError: {image:?} {reason:?}")]
pub struct IncompatibleImageError<TInput> {
    pub image: TInput,
    #[allow(dead_code)]
    pub(crate) reason: IncompatibleImageErrorReason,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum IncompatibleImageErrorReason {
    // Todo: Rename me to something more descriptive
    Comptime {
        pixel_dimensions: NonZeroU8,
        pixel_kind: &'static str,
        buffer_dimensions: NonZeroUsize,
    },
    MixedImageSizes {
        a: (NonZeroU32, NonZeroU32),
        b: (NonZeroU32, NonZeroU32),
    },
    RequiresMoreChannels {
        expected: NonZeroU8,
        actual: NonZeroU8,
    },
}

impl<'a, T: PixelType, const CHANNELS: usize> TryFrom<DynamicImageView<'a>>
    for ImageView<'a, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError<DynamicImageView<'a>>;

    fn try_from(value: DynamicImageView<'a>) -> Result<Self, Self::Error> {
        let mut value = value.channels.into_iter();
        let mut incompatible_image = Ok(());
        let mut prev_image_size = None;

        let all: [_; CHANNELS] = std::array::from_fn(|i| {
            if incompatible_image.is_ok() {
                incompatible_image = Err((
                    i,
                    match value.next() {
                        Some(dynamic) => match ImageChannelView::try_from(dynamic) {
                            Ok(typed) => {
                                let b = typed.dimensions();
                                match prev_image_size {
                                    Some(a) if a != b => (
                                        Some(typed.into()),
                                        IncompatibleImageErrorReason::MixedImageSizes { a, b },
                                    ),
                                    _ => {
                                        prev_image_size = Some(typed.dimensions());
                                        return MaybeUninit::new(typed);
                                    }
                                }
                            }
                            Err(dynamic) => (
                                Some(dynamic),
                                IncompatibleImageErrorReason::Comptime {
                                    pixel_dimensions: T::ELEMENTS,
                                    pixel_kind: std::any::type_name::<T>(),
                                    buffer_dimensions: const { NonZeroUsize::new(CHANNELS).unwrap() },
                                },
                            ),
                        },
                        None => (
                            None,
                            IncompatibleImageErrorReason::RequiresMoreChannels {
                                expected: const { crate::unwrap_usize_to_nonzero_u8(CHANNELS) },
                                actual: crate::unwrap_usize_to_nonzero_u8(i),
                            },
                        ),
                    },
                ));
            }
            MaybeUninit::uninit()
        });
        match incompatible_image {
            Ok(()) => Ok(all
                .map(|x| unsafe { x.assume_init() })
                .try_into()
                .expect("Preconditions are checked already")),
            Err((initialized_indices, (error_image, reason))) => Err(IncompatibleImageError {
                image: DynamicImageView {
                    channels: all
                        .into_iter()
                        .take(initialized_indices)
                        .map(|x| unsafe { x.assume_init() }.into())
                        .chain(error_image)
                        .chain(value)
                        .collect(),
                },
                reason,
            }),
        }
    }
}

impl<'a> DynamicImageView<'a> {
    /// Casts the first `CHANNELS` channels and checks, that their dimensions match
    fn try_cast_channels<T: PixelType, const CHANNELS: usize>(
        &self,
    ) -> Result<[&ImageChannelView<'a, T>; CHANNELS], IncompatibleImageErrorReason>
    where
        T::Primitive: Pod,
    {
        let mut result = Ok(());
        let mut iter = self.channels.iter();
        let channels = std::array::from_fn(|i| {
            if result.is_ok() {
                result = if let Some(next) = iter.next() {
                    match next.try_cast::<T>() {
                        Some(typed) => {
                            return MaybeUninit::new(typed);
                        }
                        None => Err(IncompatibleImageErrorReason::Comptime {
                            pixel_dimensions: T::ELEMENTS,
                            pixel_kind: std::any::type_name::<T>(),
                            buffer_dimensions: const { NonZeroUsize::new(CHANNELS).unwrap() },
                        }),
                    }
                } else {
                    Err(IncompatibleImageErrorReason::RequiresMoreChannels {
                        expected: const { crate::unwrap_usize_to_nonzero_u8(CHANNELS) },
                        actual: crate::unwrap_usize_to_nonzero_u8(i),
                    })
                }
            }
            MaybeUninit::uninit()
        });
        result?;
        // Safety: All channels are initialized, as result is Ok
        let channels = channels.map(|x| unsafe { x.assume_init() });
        crate::image::check_dimensions(channels.iter().map(|x| x.dimensions()))?;
        Ok(channels)
    }
}

/// Borrows the channels without copying. Mutable access (e.g. [`ImageView::make_mut`]) copies
/// the buffers, the `DynamicImageView` stays unchanged.
impl<'a, 'b, T: PixelType, const CHANNELS: usize> TryFrom<&'b DynamicImageView<'a>>
    for ImageView<'b, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError<&'b DynamicImageView<'a>>;

    fn try_from(value: &'b DynamicImageView<'a>) -> Result<Self, Self::Error> {
        match value.try_cast_channels::<T, CHANNELS>() {
            Ok(channels) => Ok(ImageView::from_channels_unchecked(
                channels.map(ImageChannelView::view),
            )),
            Err(reason) => Err(IncompatibleImageError {
                image: value,
                reason,
            }),
        }
    }
}

/// Mutably borrows the channels without copying. Mutable access (e.g. [`ImageView::make_mut`])
/// writes into the buffers of the `DynamicImageView`.
impl<'a, 'b, T: PixelType, const CHANNELS: usize> TryFrom<&'b mut DynamicImageView<'a>>
    for ImageView<'b, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError<&'b mut DynamicImageView<'a>>;

    fn try_from(value: &'b mut DynamicImageView<'a>) -> Result<Self, Self::Error> {
        // Validate with a shared borrow, so `value` can be returned in the error
        if let Err(reason) = value.try_cast_channels::<T, CHANNELS>() {
            return Err(IncompatibleImageError {
                image: value,
                reason,
            });
        }
        let mut iter = value.channels.iter_mut();
        let channels = std::array::from_fn(|_| {
            iter.next()
                .and_then(DynamicImageChannelView::try_cast_mut)
                .expect("Channels are checked above")
                .view_mut()
        });
        Ok(ImageView::from_channels_unchecked(channels))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::{Image, ImageChannel, ImageView};

    #[test]
    fn image_with_other_dimensions_is_not_uniform() {
        let ch1 = ImageChannel::<u8>::new_vec(
            vec![1, 1],
            NonZeroU32::MIN,
            const { NonZeroU32::new(2).unwrap() },
        );
        let ch2 = ImageChannel::<u8>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from_channels(ch1.into(), [ch2.into()]);
        assert!(!dynamic.is_uniform());
    }

    #[test]
    fn image_with_other_primitive_size_is_not_uniform() {
        let ch1 = ImageChannel::<u8>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        let ch2 = ImageChannel::<u16>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from_channels(ch1.into(), [ch2.into()]);
        assert!(!dynamic.is_uniform());
    }

    #[test]
    fn image_as_dynamic_is_uniform() {
        let image = Image::<u8, 3>::new_vec(
            vec![0, 1, 2, 3, 4, 5],
            NonZeroU32::MIN,
            const { NonZeroU32::new(2).unwrap() },
        );
        let dynamic = DynamicImage::from(image);
        assert!(dynamic.is_uniform());
    }

    #[test]
    fn first_mapped_returns_ref() {
        let ch = Image::<u8, 3>::new_vec(vec![1, 2, 3], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from(ch);
        let (first, mut rest) = dynamic.mapped_channels(DynamicImageChannel::buffer_flat_bytes);
        assert_eq!(1, first[0]);
        assert_eq!(2, rest.next().unwrap()[0]);
        assert_eq!(3, rest.next().unwrap()[0]);
    }

    #[test]
    fn first_mapped_returns_mut() {
        let ch = Image::<u8, 3>::new_vec(vec![1, 2, 3], NonZeroU32::MIN, NonZeroU32::MIN);
        let mut dynamic = DynamicImage::from(ch);
        let (first, mut rest) = dynamic.mapped_channels_mut(|x| x.make_mut_flat::<u8>().unwrap());
        first[0] = 4;
        assert_eq!(2, rest.next().unwrap()[0]);
        assert_eq!(3, rest.next().unwrap()[0]);
        assert_eq!(dynamic.first().buffer_flat::<u8>(), Some(&[4][..]));
    }

    #[test]
    fn buffer_flat_checks_primitive_size() {
        let channel = DynamicImageChannel::new_vec(
            vec![1u16, 2],
            NonZeroU32::MIN,
            NonZeroU32::MIN,
            const { NonZeroU8::new(2).unwrap() },
        );
        assert_eq!(channel.primitive_size().get(), 2);
        assert_eq!(channel.len_flat(), 2);
        assert_eq!(channel.buffer_flat::<u16>(), Some(&[1, 2][..]));
        assert_eq!(channel.buffer_flat::<u8>(), None);
        assert_eq!(channel.buffer_flat::<f32>(), None);
        assert_eq!(channel.buffer_flat_bytes(), &[1, 0, 2, 0]);
    }

    #[test]
    fn make_mut_flat_copies_shared_buffer() {
        let data = std::sync::Arc::<[u8]>::from([1].as_slice());
        let image = Image::<u8, 1>::new_arc(data, NonZeroU32::MIN, NonZeroU32::MIN);
        let data_ptr = image.buffer().as_ptr();
        let mut dynamic = DynamicImage::from(image);
        let clone = dynamic.clone();

        assert_eq!(dynamic[0].make_mut_flat::<u16>(), None);
        let buffer = dynamic[0].make_mut_flat::<u8>().unwrap();
        buffer[0] = 2;
        assert_ne!(buffer.as_ptr(), data_ptr);
        assert_eq!(clone.first().buffer_flat::<u8>(), Some(&[1][..]));
    }

    #[test]
    fn borrow_from_mut_dynamic_image() {
        let pixel_data = vec![1];
        let pixel_data_ptr = pixel_data.as_ptr() as usize;
        let mut image = DynamicImage::from(Image::<u8, 1>::new_vec(
            pixel_data,
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
        let mut image_back: ImageView<'_, u8, 1> = (&mut image).try_into().unwrap();

        assert_eq!(pixel_data_ptr, image_back.make_mut()[0].as_ptr() as usize);
        image_back.make_mut()[0][0] = 2;
        drop(image_back);
        assert_eq!(image.first().buffer_flat::<u8>(), Some(&[2][..]));
    }

    #[test]
    fn borrow_from_mut_dynamic_image_copies_shared_buffer_into_dynamic_image() {
        let data = std::sync::Arc::<[u8]>::from([1].as_slice());
        let data_ptr = data.as_ptr();
        let mut dynamic = DynamicImage::from(Image::<u8, 1>::new_arc(
            data,
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
        let clone = dynamic.clone();

        let mut view = ImageView::<u8, 1>::try_from(&mut dynamic).unwrap();
        assert_eq!(view.buffer().as_ptr(), data_ptr, "Reading doesn't copy");
        view.make_mut_flat()[0] = 2;
        assert_ne!(view.buffer().as_ptr(), data_ptr);

        let view_clone = view.clone();
        view.make_mut_flat()[0] = 3;
        assert_eq!(view_clone.buffer(), &[2], "Clones of mutable views copy");
        drop((view, view_clone));

        assert_eq!(dynamic.first().buffer_flat::<u8>(), Some(&[3][..]));
        assert_eq!(clone.first().buffer_flat::<u8>(), Some(&[1][..]));
    }

    #[test]
    fn borrow_from_dynamic_image_copies_on_write() {
        let dynamic = DynamicImage::from(Image::<u8, 2>::new_vec(
            vec![1, 2],
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
        let mut view = ImageView::<u8, 2>::try_from(&dynamic).unwrap();
        assert_eq!(
            view.buffers()[1].as_ptr(),
            dynamic[1].buffer_flat::<u8>().unwrap().as_ptr()
        );
        let [_, second] = view.make_mut();
        second[0] = 3;
        assert_eq!(view.buffers(), [[1], [3]]);
        assert_eq!(dynamic[1].buffer_flat::<u8>(), Some(&[2][..]));
    }

    #[test]
    fn borrowed_image_into_dynamic_image_and_back() {
        let mut data = [1u16, 2];
        let data_ptr = data.as_ptr();
        let view = ImageView::<u16, 2>::new_slice_mut(&mut data, NonZeroU32::MIN, NonZeroU32::MIN);
        let mut dynamic = DynamicImageView::from(view);
        dynamic[1].make_mut_flat::<u16>().unwrap()[0] = 3;
        let back = ImageView::<u16, 2>::try_from(dynamic).unwrap();
        assert_eq!(back.buffers()[0].as_ptr(), data_ptr);
        drop(back);
        assert_eq!(data, [1, 3]);
    }

    #[test]
    fn borrow_from_different_size_mut_dynamic_image() {
        let mut dynamic = DynamicImage::from(Image::<u8, 2>::new_vec(
            vec![0, 1],
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
        dynamic[1] = ImageChannel::<u8>::new_vec(
            vec![0, 1],
            NonZeroU32::MIN,
            const { NonZeroU32::new(2).unwrap() },
        )
        .into();

        ImageView::<u8, 2>::try_from(&mut dynamic).unwrap_err();
        ImageView::<u8, 2>::try_from(&dynamic).unwrap_err();
    }

    #[test]
    fn create_from_luma_u8() {
        let luma = Image::<u8, 1>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from(luma);
        assert_eq!(1, dynamic.channels.len());
        let luma_back: Image<u8, 1> = dynamic.try_into().unwrap();
        assert_eq!(luma_back.into_vec(), vec![1]);
    }

    #[test]
    fn create_from_luma_rgb16_interleaved() {
        let luma =
            Image::<[u16; 3], 1>::new_vec(vec![[1u16, 2, 3]], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from(luma);
        assert_eq!(1, dynamic.channels.len());
        let luma_back: Image<[u16; 3], 1> = dynamic.try_into().unwrap();
        assert_eq!(luma_back.into_vec(), vec![[1u16, 2, 3]]);
    }

    #[test]
    fn create_from_rgb8_interleaved() {
        let rgb = Image::<[u8; 3], 1>::new_vec(vec![[1u8, 2, 3]], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from(rgb);
        assert_eq!(1, dynamic.channels.len());
        assert_eq!(1, dynamic[0].primitive_size().get());
        assert_eq!(3, dynamic[0].pixel_elements().get());
        let rgb_back: Image<[u8; 3], 1> = dynamic.try_into().unwrap();
        assert_eq!(rgb_back.into_vec(), vec![[1u8, 2, 3]]);
    }
    #[test]
    fn create_from_luma_rgb8_planar() {
        let luma = Image::<u8, 3>::new_vec(vec![1u8, 2, 3], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from(luma);
        assert_eq!(3, dynamic.channels.len());
        let luma_back: Image<u8, 3> = dynamic.try_into().unwrap();
        assert_eq!(luma_back.into_vec(), vec![1u8, 2, 3]);
    }

    #[test]
    fn clone_dynamic_image() {
        let width = NonZeroU32::new(2).unwrap();
        let height = NonZeroU32::new(2).unwrap();
        let luma = Image::<u8, 1>::new_vec(vec![1, 2, 3, 4], width, height);
        let dynamic = DynamicImage::from(luma);
        let cloned = dynamic.clone();

        // Verify both can be converted back to the same image
        let luma_back: Image<u8, 1> = dynamic.try_into().unwrap();
        {
            let ref_luma: ImageView<u8, 1> = (&cloned).try_into().unwrap();
            assert_eq!(ref_luma.dimensions(), (width, height));
        }
        let luma_cloned: Image<u8, 1> = cloned.try_into().unwrap();
        let vec_back = luma_back.into_vec();
        let vec_cloned = luma_cloned.into_vec();
        assert_eq!(vec_back, vec_cloned);
        assert_eq!(vec_cloned, vec![1, 2, 3, 4]);
    }

    #[test]
    fn create_from_incompatible_image() {
        let luma = Image::<u8, 1>::new_vec(vec![42], NonZeroU32::MIN, NonZeroU32::MIN);
        let dynamic = DynamicImage::from(luma.clone());
        let incompatible = Image::<u16, 1>::try_from(dynamic).unwrap_err();
        assert_eq!(incompatible.image, DynamicImage::from(luma));
    }
}
