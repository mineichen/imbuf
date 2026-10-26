use std::{
    fmt::{self, Debug, Formatter},
    iter::{self, once},
    mem::MaybeUninit,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    ops::Deref,
    ptr, slice, vec,
};

use bytemuck::Pod;

use crate::{
    Image, ImageChannel, ImageMut, ImageRef, PixelType, PixelTypePrimitive,
    channel::UnsafeImageChannel,
};

/// Image with number of channels and their types and dimensions only known at runtime
/// There are no guarantees that the types or dimensions of channels match. See `ImageChannels` for more information.
/// The public interface is designed, so it can be extended to support images, which cannot be represented with Image (e.g. 1 Channel u8 and the other f32)
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicImage {
    channels: Vec<DynamicImageChannel>,
}

impl DynamicImage {
    pub fn from_channels(
        first: DynamicImageChannel,
        rest: impl IntoIterator<Item = DynamicImageChannel>,
    ) -> Self {
        Self {
            channels: once(first).chain(rest).collect(),
        }
    }
    /// `DynamicImage` always has at least one channel, so this never panics
    #[must_use]
    pub fn first(&self) -> &DynamicImageChannel {
        &self.channels[0]
    }

    /// `DynamicImage` always has at least one channel, so this never panics
    #[must_use]
    pub fn last(&self) -> &DynamicImageChannel {
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
    pub fn mapped_channels<'a, T: 'a, TF>(
        &'a self,
        mapper: TF,
    ) -> (T, iter::Map<slice::Iter<'a, DynamicImageChannel>, TF>)
    where
        TF: FnMut(&'a DynamicImageChannel) -> T,
    {
        let mut iter = self.iter().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn mapped_channels_mut<'a, T: 'a, TF>(
        &'a mut self,
        mapper: TF,
    ) -> (T, iter::Map<slice::IterMut<'a, DynamicImageChannel>, TF>)
    where
        TF: FnMut(&'a mut DynamicImageChannel) -> T,
    {
        let mut iter = self.iter_mut().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn into_mapped_channels<T, TF>(
        self,
        mapper: TF,
    ) -> (T, iter::Map<vec::IntoIter<DynamicImageChannel>, TF>)
    where
        TF: FnMut(DynamicImageChannel) -> T,
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

// Deref slice only, to make sure, noone can create a DynamicImage with a empty Vec
impl Deref for DynamicImage {
    type Target = [DynamicImageChannel];

    fn deref(&self) -> &Self::Target {
        &self.channels
    }
}

impl IntoIterator for DynamicImage {
    type Item = DynamicImageChannel;
    type IntoIter = vec::IntoIter<DynamicImageChannel>;

    fn into_iter(self) -> Self::IntoIter {
        self.channels.into_iter()
    }
}

impl std::ops::DerefMut for DynamicImage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.channels
    }
}

/// Channel whose primitive and number of pixel elements are only known at runtime.
///
/// The channel only knows the size of its primitive (e.g. 2 bytes for `u16`). Its buffer can
/// therefore be viewed as any [`Pod`] primitive of the same size and alignment, e.g. via
/// [`DynamicImageChannel::buffer_flat`], [`DynamicImageChannel::try_cast`] or
/// `ImageChannel::try_from`. For the same reason, only [`ImageChannel`]s of `Pod` primitives can
/// be converted into a `DynamicImageChannel`:
///
/// ```compile_fail,E0277
/// use std::num::{NonZeroU8, NonZeroU32};
/// use imbuf::{DynamicImageChannel, ImageChannel, PixelTypePrimitive};
///
/// #[derive(Clone, PartialEq)]
/// struct NonZero(NonZeroU8);
/// impl PixelTypePrimitive for NonZero {}
///
/// let channel = ImageChannel::new_vec(vec![NonZero(NonZeroU8::MIN)], NonZeroU32::MIN, NonZeroU32::MIN);
/// // Would allow writing 0 through `ImageChannel<u8>`
/// let dynamic = DynamicImageChannel::from(channel);
/// ```
#[repr(transparent)]
pub struct DynamicImageChannel(UnsafeImageChannel);

impl DynamicImageChannel {
    /// Takes ownership of an [`UnsafeImageChannel`], e.g. to support a custom storage.
    ///
    /// # Safety
    /// `channel` has to uphold the following invariants:
    /// - The size of the buffer in bytes (`width * height * pixel_elements * vtable.primitive_size()`)
    ///   doesn't exceed `isize::MAX`.
    /// - `ptr` points to an initialized buffer of that size, which stays valid until `vtable.drop`
    ///   is called. The buffer can be shared between instances.
    /// - `vtable` is able to work with `ptr` and `data` (see [`ImageChannelVTable`](crate::ImageChannelVTable)).
    ///   Any bytes written through a `Pod` view of the buffer have to be valid for the vtable
    ///   (e.g. the primitive of the vtable is `Pod` itself).
    /// - `ptr` and `data` can be sent to and shared between threads.
    #[must_use]
    pub const unsafe fn from_unsafe(channel: UnsafeImageChannel) -> Self {
        Self(channel)
    }

    /// Read access to the raw parts, e.g. to implement an [`ImageChannelVTable`](crate::ImageChannelVTable).
    #[must_use]
    pub const fn get_unsafe(channel: &Self) -> &UnsafeImageChannel {
        &channel.0
    }

    /// Write access to the raw parts, e.g. to implement [`ImageChannelVTable::make_mut`](crate::ImageChannelVTable::make_mut).
    ///
    /// # Safety
    /// The invariants of [`DynamicImageChannel::from_unsafe`] have to hold when the borrow ends.
    /// Assigning a new `UnsafeImageChannel` leaks the previous buffer, as it doesn't implement
    /// `Drop`. Assign a `DynamicImageChannel` to the outer reference instead.
    pub const unsafe fn get_unsafe_mut(channel: &mut Self) -> &mut UnsafeImageChannel {
        &mut channel.0
    }

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

    #[must_use]
    pub const fn width(&self) -> NonZeroU32 {
        self.0.width
    }

    #[must_use]
    pub const fn height(&self) -> NonZeroU32 {
        self.0.height
    }

    #[must_use]
    pub const fn dimensions(&self) -> (NonZeroU32, NonZeroU32) {
        (self.0.width, self.0.height)
    }

    #[must_use]
    pub const fn dimensions_zeroable(&self) -> (u32, u32) {
        (self.0.width.get(), self.0.height.get())
    }

    #[must_use]
    pub const fn pixel_elements(&self) -> NonZeroU8 {
        self.0.pixel_elements
    }

    /// Size of a single primitive in bytes
    #[must_use]
    pub fn primitive_size(&self) -> NonZeroUsize {
        (self.0.vtable.primitive_size)()
    }

    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.0.calc_len_packed()
    }

    /// Number of primitives in the buffer (`width * height * pixel_elements`)
    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub const fn len_flat(&self) -> usize {
        self.0.calc_len_flat()
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
        unsafe { slice::from_raw_parts(self.0.ptr.cast::<T>(), self.len_flat()) }
    }

    /// Mutable access to the buffer, which is copied if it is shared (copy on write).
    ///
    /// Returns `None`, if the size or the alignment of `T` don't match the primitive of the channel
    pub fn primitive_make_mut<T: PixelTypePrimitive + Pod>(&mut self) -> Option<&mut [T]> {
        if !self.fits_primitive::<T>() {
            return None;
        }
        Some(unsafe { self.primitive_make_mut_unchecked() })
    }

    pub(crate) unsafe fn primitive_make_mut_unchecked<T: PixelTypePrimitive>(
        &mut self,
    ) -> &mut [T] {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.0.vtable.make_mut)(self) };
        // Safety: Size and alignment are checked after make_mut, which might reallocate
        unsafe { self.buffer_flat_mut_unchecked() }
    }

    #[must_use]
    pub fn buffer_flat_bytes(&self) -> &[u8] {
        let len = self.len_flat() * self.primitive_size().get();
        // Safety: The buffer is initialized (invariant of the channel)
        unsafe { slice::from_raw_parts(self.0.ptr, len) }
    }

    /// Try to view this channel as an `ImageChannel<TP>`.
    ///
    /// Returns `None`, if `pixel_elements`, the size or the alignment of `TP::Primitive` don't match.
    #[must_use]
    pub fn try_cast<TP: PixelType>(&self) -> Option<&ImageChannel<TP>>
    where
        TP::Primitive: Pod,
    {
        // Safety: `ImageChannel` is repr(transparent) over `DynamicImageChannel`. The layout is
        // checked and Pod accepts any bit pattern
        self.fits_pixel::<TP>()
            .then(|| unsafe { &*ptr::from_ref(self).cast::<ImageChannel<TP>>() })
    }

    /// Mutable version of [`DynamicImageChannel::try_cast`]
    #[must_use]
    pub fn try_cast_mut<TP: PixelType>(&mut self) -> Option<&mut ImageChannel<TP>>
    where
        TP::Primitive: Pod,
    {
        // Safety: See try_cast
        self.fits_pixel::<TP>()
            .then(|| unsafe { &mut *ptr::from_mut(self).cast::<ImageChannel<TP>>() })
    }

    /// Checks, if the buffer can be viewed as `T`
    pub(crate) fn fits_primitive<T>(&self) -> bool {
        self.primitive_size().get() == size_of::<T>() && self.0.ptr.cast::<T>().is_aligned()
    }

    /// Checks, if the buffer can be viewed as `TP`
    pub(crate) fn fits_pixel<TP: PixelType>(&self) -> bool {
        self.pixel_elements() == TP::ELEMENTS && self.fits_primitive::<TP::Primitive>()
    }

    /// # Safety
    /// See `buffer_flat_unchecked`. Additionally, the buffer must be exclusive (`make_mut`)
    pub(crate) unsafe fn buffer_flat_mut_unchecked<T>(&mut self) -> &mut [T] {
        debug_assert!(self.fits_primitive::<T>());
        // Safety: Forwarded to the caller
        unsafe { slice::from_raw_parts_mut(self.0.ptr.cast::<T>().cast_mut(), self.len_flat()) }
    }
}

impl Drop for DynamicImageChannel {
    fn drop(&mut self) {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.0.vtable.drop)(self) };
    }
}

impl Clone for DynamicImageChannel {
    fn clone(&self) -> Self {
        // Safety: The vtable is able to work with the channel
        unsafe { (self.0.vtable.clone)(self) }
    }
}

/// Channels are equal, if their dimensions, `pixel_elements`, `primitive_size` and bytes are equal
impl PartialEq for DynamicImageChannel {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions() == other.dimensions()
            && self.pixel_elements() == other.pixel_elements()
            && self.primitive_size() == other.primitive_size()
            && self.buffer_flat_bytes() == other.buffer_flat_bytes()
    }
}

impl Debug for DynamicImageChannel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicImageChannel")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("primitive_size", &self.primitive_size())
            .field("pixel_elements", &self.pixel_elements())
            .finish()
    }
}

// Safety: The primitives are Send + Sync and storages have to be thread safe (invariant of `from_unsafe`)
unsafe impl Send for DynamicImageChannel {}
unsafe impl Sync for DynamicImageChannel {}

impl<TPixel: PixelType, const CHANNELS: usize> From<Image<TPixel, CHANNELS>> for DynamicImage
where
    TPixel::Primitive: Pod,
{
    fn from(value: Image<TPixel, CHANNELS>) -> Self {
        DynamicImage {
            channels: <[ImageChannel<TPixel>; CHANNELS]>::from(value)
                .into_iter()
                .map(ImageChannel::into)
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

impl<T: PixelType, const CHANNELS: usize> TryFrom<DynamicImage> for Image<T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError<DynamicImage>;

    fn try_from(value: DynamicImage) -> Result<Self, Self::Error> {
        let mut value = value.channels.into_iter();
        let mut incompatible_image = Ok(());
        let mut prev_image_size = None;

        let all: [_; CHANNELS] = std::array::from_fn(|i| {
            if incompatible_image.is_ok() {
                incompatible_image = Err((
                    i,
                    match value.next() {
                        Some(dynamic) => match ImageChannel::try_from(dynamic) {
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
                image: DynamicImage {
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

impl<'a, T: PixelType, const CHANNELS: usize> TryFrom<&'a DynamicImage>
    for ImageRef<'a, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError<&'a DynamicImage>;

    fn try_from(value: &'a DynamicImage) -> Result<Self, Self::Error> {
        let mut result = Ok(());
        let mut iter = value.channels.iter();
        let channels = std::array::from_fn(|i| {
            if result.is_ok() {
                result = if let Some(next) = iter.next() {
                    match <&ImageChannel<T>>::try_from(next) {
                        Ok(typed) => {
                            return MaybeUninit::new(typed);
                        }
                        Err(()) => Err(IncompatibleImageErrorReason::Comptime {
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
        result
            .and_then(|()| {
                <ImageRef<'a, T, CHANNELS>>::try_from(channels.map(|x| unsafe { x.assume_init() }))
                    .map_err(|x: IncompatibleImageError<[&ImageChannel<T>; CHANNELS]>| x.reason)
            })
            .map_err(|reason| IncompatibleImageError {
                image: value,
                reason,
            })
    }
}

impl<'a, T: PixelType, const CHANNELS: usize> TryFrom<&'a mut DynamicImage>
    for ImageMut<'a, T, CHANNELS>
where
    T::Primitive: Pod,
{
    type Error = IncompatibleImageError<&'a mut DynamicImage>;

    fn try_from(value: &'a mut DynamicImage) -> Result<Self, Self::Error> {
        // Validate with a shared borrow, so `value` can be returned in the error
        if let Err(e) = ImageRef::<T, CHANNELS>::try_from(&*value) {
            let reason = e.reason;
            return Err(IncompatibleImageError {
                image: value,
                reason,
            });
        }
        let mut iter = value.channels.iter_mut();
        let channels = std::array::from_fn(|_| {
            iter.next()
                .and_then(DynamicImageChannel::try_cast_mut)
                .expect("Channels are checked above")
        });
        Ok(ImageMut::try_from(channels).expect("Dimensions are checked above"))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

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
        let (first, mut rest) =
            dynamic.mapped_channels_mut(|x| x.primitive_make_mut::<u8>().unwrap());
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
    fn primitive_make_mut_copies_shared_buffer() {
        let data = std::sync::Arc::<[u8]>::from([1].as_slice());
        let image = Image::<u8, 1>::new_arc(data, NonZeroU32::MIN, NonZeroU32::MIN);
        let data_ptr = image.buffer().as_ptr();
        let mut dynamic = DynamicImage::from(image);
        let clone = dynamic.clone();

        assert_eq!(dynamic[0].primitive_make_mut::<u16>(), None);
        let buffer = dynamic[0].primitive_make_mut::<u8>().unwrap();
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
        let mut image_back: ImageMut<'_, u8, 1> = (&mut image).try_into().unwrap();

        assert_eq!(pixel_data_ptr, image_back.make_mut()[0].as_ptr() as usize);
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

        ImageMut::<u8, 2>::try_from(&mut dynamic).unwrap_err();
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
            let ref_luma: ImageRef<u8, 1> = (&cloned).try_into().unwrap();
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
