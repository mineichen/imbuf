use std::{
    fmt::{self, Debug, Formatter},
    mem::MaybeUninit,
    num::{NonZeroU8, NonZeroU32},
    ptr,
};

use crate::{
    DynamicImageChannelView, IncompatibleImage, InvalidLenError,
    channel::ImageChannelView,
    error::IncompatibleImageError,
    pixel::{PixelType, PixelTypePrimitive},
    shared_vec, transpose, unwrap_usize_to_nonzero_u8,
};

/// [`ImageView`] which owns its buffers (or shares them with other owners)
pub type Image<T, const CHANNELS: usize> = ImageView<'static, T, CHANNELS>;

/// Represents a image, where all channels share the same width, height. You usually want to use
/// its typedef versions [`Image`] (owned) and [`ImageView`] if lifetime is not `'static` instead.
///
/// The buffers are valid for the lifetime `'a`, see [`ImageChannelView`]. Borrowed images are
/// created with [`ImageView::new_slice`], [`ImageView::view`], `TryFrom<&DynamicImageView>`
/// and their `mut` counterparts.
#[derive(Clone)]
#[repr(transparent)]
pub struct ImageView<'a, T: PixelType, const CHANNELS: usize>([ImageChannelView<'a, T>; CHANNELS]);

impl<T: PixelType, const CHANNELS: usize> PartialEq for ImageView<'_, T, CHANNELS> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

impl<T: PixelType, const CHANNELS: usize> ImageView<'_, T, CHANNELS> {
    /// Checks that `width * height * CHANNELS` doesn't overflow `usize` and matches `len`, e.g. to
    /// validate untrusted dimensions for a buffer of `T` with all channels (like the buffer of
    /// [`Image::new_vec`] or [`ImageView::new_slice`]).
    pub const fn ensure_valid_len(
        width: NonZeroU32,
        height: NonZeroU32,
        len: usize,
    ) -> Result<(), InvalidLenError> {
        let channels = const { unwrap_usize_to_nonzero_u8(CHANNELS) };
        DynamicImageChannelView::ensure_len(width, height, NonZeroU8::MIN, channels, len)
    }

    /// Checks that `width * height * T::ELEMENTS * CHANNELS` doesn't overflow `usize` and matches
    /// `len`, e.g. to validate untrusted dimensions for a buffer of primitives with all channels
    /// (like the buffer of [`Image::new_vec_flat`]).
    pub const fn ensure_valid_flat_len(
        width: NonZeroU32,
        height: NonZeroU32,
        len: usize,
    ) -> Result<(), InvalidLenError> {
        let channels = const { unwrap_usize_to_nonzero_u8(CHANNELS) };
        DynamicImageChannelView::ensure_len(width, height, T::ELEMENTS, channels, len)
    }
}

#[allow(clippy::len_without_is_empty)]
impl<'a, T: PixelType, const CHANNELS: usize> ImageView<'a, T, CHANNELS> {
    /// Borrows the planar `input` (all pixels of the first channel, then the second...) without
    /// copying. Mutable access (e.g. [`ImageView::make_mut`]) copies the buffers, `input` stays
    /// unchanged.
    pub const fn new_slice(
        input: &'a [T],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        match Self::ensure_valid_len(width, height, input.len()) {
            // Safety: The size is checked
            Ok(()) => Ok(unsafe { Self::new_slice_unchecked(input, width, height) }),
            Err(e) => Err(e),
        }
    }

    /// Like [`ImageView::new_slice`], but panics on an invalid length. Usable in const contexts,
    /// where the `Result` of [`ImageView::new_slice`] can't be unwrapped, as the image has a
    /// destructor:
    ///
    /// ```
    /// use std::num::NonZeroU32;
    /// use imbuf::Image;
    ///
    /// const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    /// const IMAGE: Image<u8, 2> = Image::new_slice_unwrap(&[1, 2, 3, 4], NZ_2, NonZeroU32::MIN);
    /// assert_eq!(IMAGE.buffers(), [[1, 2], [3, 4]]);
    /// ```
    ///
    /// # Panics
    /// If `width * height * CHANNELS` overflows `usize` or doesn't match `input.len()`
    #[must_use]
    pub const fn new_slice_unwrap(input: &'a [T], width: NonZeroU32, height: NonZeroU32) -> Self {
        assert!(
            Self::ensure_valid_len(width, height, input.len()).is_ok(),
            "The length of the slice doesn't match width * height * CHANNELS"
        );
        // Safety: The size is checked
        unsafe { Self::new_slice_unchecked(input, width, height) }
    }

    /// # Safety
    /// `width * height * CHANNELS` mustn't overflow `usize` and has to match `input.len()`
    const unsafe fn new_slice_unchecked(
        input: &'a [T],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Self {
        let channel_len = input.len() / CHANNELS;
        let mut channels = [const { MaybeUninit::<ImageChannelView<'a, T>>::uninit() }; CHANNELS];
        let mut rest = input;
        let mut i = 0;
        while i < CHANNELS {
            let (chunk, tail) = rest.split_at(channel_len);
            // Safety: Each chunk has `width * height` elements (precondition)
            channels[i]
                .write(unsafe { ImageChannelView::new_slice_unchecked(chunk, width, height) });
            rest = tail;
            i += 1;
        }
        // Safety: All channels are initialized. `MaybeUninit<X>` has the layout of `X`
        Self(unsafe {
            channels
                .as_ptr()
                .cast::<[ImageChannelView<'a, T>; CHANNELS]>()
                .read()
        })
    }

    /// Mutably borrows the planar `input` without copying. Mutable access
    /// (e.g. [`ImageView::make_mut`]) writes into `input`. Clones copy the buffers.
    #[allow(clippy::missing_panics_doc)]
    pub fn new_slice_mut(
        input: &'a mut [T],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        let len = input.len();
        Self::ensure_valid_len(width, height, len)?;
        let mut chunks = input.chunks_exact_mut(len / CHANNELS);
        Ok(Self(std::array::from_fn(|_| {
            let chunk = chunks.next().expect("Length is checked");
            ImageChannelView::new_slice_mut(chunk, width, height).expect("Length is checked")
        })))
    }

    /// Borrows all channels without copying the buffers, see [`ImageChannelView::view`]
    #[must_use]
    pub fn view(&self) -> ImageView<'_, T, CHANNELS> {
        ImageView(self.0.each_ref().map(ImageChannelView::view))
    }

    /// Mutably borrows all channels without copying the buffers, see [`ImageChannelView::view_mut`]
    #[must_use]
    pub fn view_mut(&mut self) -> ImageView<'_, T, CHANNELS> {
        ImageView(self.0.each_mut().map(ImageChannelView::view_mut))
    }

    /// The caller has to make sure, that all channels have the same dimensions
    pub(crate) const fn from_channels_unchecked(
        channels: [ImageChannelView<'a, T>; CHANNELS],
    ) -> Self {
        Self(channels)
    }

    /// Views the channels of a [`DynamicImageView`](crate::DynamicImageView) as an image without
    /// copying
    ///
    /// # Safety
    /// All channels have to fit `T` (see [`ImageChannelView::from_dynamic_unchecked`]) and have
    /// the same dimensions
    pub(crate) unsafe fn from_dynamic_ref_unchecked<'b>(
        channels: &'b [DynamicImageChannelView<'a>; CHANNELS],
    ) -> &'b Self {
        debug_assert_eq!(Ok(()), Self::ensure_fits_dynamic(channels));
        // Safety: `ImageView` is repr(transparent) over an array of `ImageChannelView`, which is
        // repr(transparent) over `DynamicImageChannelView`. The layout is forwarded to the caller
        unsafe { &*ptr::from_ref(channels).cast::<Self>() }
    }

    /// Mutable reference version of [`ImageView::from_dynamic_ref_unchecked`]
    ///
    /// # Safety
    /// See [`ImageView::from_dynamic_ref_unchecked`]
    pub(crate) unsafe fn from_dynamic_mut_unchecked<'b>(
        channels: &'b mut [DynamicImageChannelView<'a>; CHANNELS],
    ) -> &'b mut Self {
        debug_assert_eq!(Ok(()), Self::ensure_fits_dynamic(channels));
        // Safety: See `from_dynamic_ref_unchecked`
        unsafe { &mut *ptr::from_mut(channels).cast::<Self>() }
    }

    /// Checks the requirements of [`ImageView::from_dynamic_ref_unchecked`]
    fn ensure_fits_dynamic(
        channels: &[DynamicImageChannelView<'a>; CHANNELS],
    ) -> Result<(), IncompatibleImageError> {
        channels
            .iter()
            .try_for_each(DynamicImageChannelView::ensure_fits_pixel::<T>)?;
        check_dimensions(channels.iter().map(DynamicImageChannelView::dimensions))
    }

    #[must_use]
    pub fn into_channels(self) -> [ImageChannelView<'a, T>; CHANNELS] {
        self.0
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn into_vec(self) -> Vec<T> {
        if CHANNELS == 1 {
            // For single channel, use ImageChannel::into_vec which preserves pointer reuse
            let channel = self.0.into_iter().next().expect("CHANNELS == 1");
            channel.into_vec()
        } else {
            // For multiple channels, concatenate them
            let mut result = Vec::with_capacity(self.len_per_channel() * CHANNELS);
            for channel in self.0 {
                result.extend_from_slice(channel.buffer());
            }
            result
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn make_mut(&mut self) -> [&mut [T]; CHANNELS] {
        let mut iter = self.0.iter_mut();
        std::array::from_fn(|_| iter.next().unwrap().make_mut())
    }

    #[must_use]
    pub fn width(&self) -> NonZeroU32 {
        // All channels have the same height (validated at construction)
        // CHANNELS is always > 0
        self.0[0].width()
    }

    #[must_use]
    pub fn height(&self) -> NonZeroU32 {
        // All channels have the same height (validated at construction)
        // CHANNELS is always > 0
        self.0[0].height()
    }

    #[must_use]
    pub fn dimensions(&self) -> (NonZeroU32, NonZeroU32) {
        // All channels have the same height (validated at construction)
        // CHANNELS is always > 0
        self.0[0].dimensions()
    }

    #[must_use]
    pub fn dimensions_zeroable(&self) -> (u32, u32) {
        self.0[0].dimensions_zeroable()
    }

    /// Returns the number of pixels in each image channel
    #[must_use]
    pub fn len_per_channel(&self) -> usize {
        self.0[0].len()
    }

    #[must_use]
    pub fn len_per_channel_flat(&self) -> usize {
        self.0[0].len_flat()
    }

    #[must_use]
    pub fn buffers(&self) -> [&[T]; CHANNELS] {
        self.0.each_ref().map(ImageChannelView::buffer)
    }

    /// Mutable access to the buffers without copying them, see [`ImageChannelView::buffer_mut`].
    /// `None`, if any channel requires a copy. [`ImageView::make_mut`] copies in that case.
    pub fn buffers_mut(&mut self) -> Option<[&mut [T]; CHANNELS]> {
        // `try_make_mut` doesn't share the buffer of a channel, so the channels checked before
        // stay exclusive
        if self.0.iter_mut().all(ImageChannelView::try_make_exclusive) {
            // Safety: All buffers are exclusive
            Some(
                self.0
                    .each_mut()
                    .map(|channel| unsafe { channel.buffer_mut_unchecked() }),
            )
        } else {
            None
        }
    }

    /// Borrows the first channel (the single channel of a one-channel image).
    #[must_use]
    pub fn first(&self) -> &ImageChannelView<'a, T> {
        &self.0[0]
    }

    /// Mutably borrows the first channel (the single channel of a one-channel image).
    #[must_use]
    pub fn first_mut(&mut self) -> &mut ImageChannelView<'a, T> {
        &mut self.0[0]
    }
}

impl<'a, T: PixelType> ImageView<'a, T, 1> {
    #[must_use]
    pub fn from_channel(ch: ImageChannelView<'a, T>) -> Self {
        Self([ch])
    }

    #[must_use]
    pub fn buffer(&self) -> &[T] {
        self.0[0].buffer()
    }

    #[must_use]
    pub fn buffer_flat(&self) -> &[T::Primitive] {
        self.0[0].buffer_flat()
    }

    pub fn make_mut_flat(&mut self) -> &mut [T::Primitive] {
        self.0[0].make_mut_flat()
    }

    /// Mutable access to the buffer without copying it, see [`ImageChannelView::buffer_mut`]
    pub fn buffer_mut(&mut self) -> Option<&mut [T]> {
        self.0[0].buffer_mut()
    }

    /// Flat version of [`ImageView::buffer_mut`]
    pub fn buffer_flat_mut(&mut self) -> Option<&mut [T::Primitive]> {
        self.0[0].buffer_flat_mut()
    }
}

impl<T: PixelTypePrimitive, const CHANNELS: usize> ImageView<'_, T, CHANNELS> {
    /// Interleaves the planar channels into a single channel with one element per channel in
    /// each pixel (e.g. planar RGB into `[u8; 3]` pixels). `transpose` of the result restores the
    /// planar image.
    ///
    /// The buffer is transposed in place without copying, if the channels exclusively own a
    /// single planar buffer (e.g. created by [`Image::new_vec`] and not cloned). A single channel
    /// is reused, if it exclusively owns a `Vec`. Otherwise, the pixels are copied (e.g. for
    /// [`ImageView::view`]).
    ///
    /// # Panics
    /// If the channels have different numbers of pixels (e.g. after replacing a channel with
    /// [`ImageView::first_mut`]).
    #[must_use]
    pub fn transpose(self) -> Image<[T; CHANNELS], 1> {
        let (width, height) = self.dimensions();
        let interleaved = if CHANNELS == 1 {
            // A single channel is planar and interleaved at the same time
            let channel = self.0.into_iter().next().expect("CHANNELS == 1");
            channel.into_vec_flat()
        } else {
            match shared_vec::try_into_vec(self.0) {
                Ok(mut buffer) => {
                    transpose::interleave_in_place::<T, CHANNELS>(&mut buffer);
                    buffer
                }
                Err(channels) => {
                    let pixels =
                        transpose::interleave(channels.each_ref().map(ImageChannelView::buffer));
                    crate::pixel::flat_vec(pixels)
                }
            }
        };
        Image::new_vec_flat(interleaved, width, height)
            .expect("The planes have `width * height` elements")
    }
}

impl<T: PixelTypePrimitive, const PIXEL_ELEMENTS: usize> ImageView<'_, [T; PIXEL_ELEMENTS], 1> {
    /// Splits the pixels into one planar channel per pixel element (e.g. `[u8; 3]` pixels into
    /// planar RGB). `transpose` of the result restores the interleaved image.
    ///
    /// The buffer is transposed in place without copying, if the channel exclusively owns a `Vec`
    /// (e.g. created by [`Image::new_vec`]). Otherwise, the pixels are copied (e.g. for
    /// [`ImageView::view`]).
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn transpose(self) -> Image<T, PIXEL_ELEMENTS> {
        let (width, height) = self.dimensions();
        let [channel] = self.0;
        let planar = match channel.try_into_vec_flat() {
            Ok(mut buffer) => {
                transpose::deinterleave_in_place::<T, PIXEL_ELEMENTS>(&mut buffer);
                buffer
            }
            Err(channel) => transpose::deinterleave(channel.buffer()),
        };
        Image::new_vec(planar, width, height).expect("The channel has `width * height` pixels")
    }
}

impl<T: PixelType, const CHANNELS: usize> Debug for ImageView<'_, T, CHANNELS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("channels", &CHANNELS)
            .field("pixel", &std::any::type_name::<T>())
            .finish()
    }
}

/// Fails with `MixedImageSizes`, if not all dimensions are equal
pub(crate) fn check_dimensions(
    mut dimensions: impl Iterator<Item = (NonZeroU32, NonZeroU32)>,
) -> Result<(), IncompatibleImageError> {
    let Some(a) = dimensions.next() else {
        return Ok(());
    };
    match dimensions.find(|b| a != *b) {
        Some(b) => Err(IncompatibleImageError::MixedImageSizes { a, b }),
        None => Ok(()),
    }
}

impl<'a, T: PixelType, const CHANNELS: usize> TryFrom<[ImageChannelView<'a, T>; CHANNELS]>
    for ImageView<'a, T, CHANNELS>
{
    type Error = IncompatibleImage<[ImageChannelView<'a, T>; CHANNELS]>;
    fn try_from(channels: [ImageChannelView<'a, T>; CHANNELS]) -> Result<Self, Self::Error> {
        let _assert_not_empty = const { unwrap_usize_to_nonzero_u8(CHANNELS) };

        match check_dimensions(channels.iter().map(ImageChannelView::dimensions)) {
            Ok(()) => Ok(Self(channels)),
            Err(error) => Err(IncompatibleImage {
                image: channels,
                error,
            }),
        }
    }
}

impl<'a, T: PixelType, const CHANNELS: usize> From<ImageView<'a, T, CHANNELS>>
    for [ImageChannelView<'a, T>; CHANNELS]
{
    fn from(value: ImageView<'a, T, CHANNELS>) -> Self {
        value.0
    }
}

/// A single channel is always a valid image, see [`ImageView::from_channel`]
impl<'a, T: PixelType> From<ImageChannelView<'a, T>> for ImageView<'a, T, 1> {
    fn from(channel: ImageChannelView<'a, T>) -> Self {
        Self::from_channel(channel)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::num::NonZeroU32;

    use testresult::TestResult;

    use super::*;
    use crate::ImageChannel;

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    const LUMA8: ImageChannel<u8> = ImageChannel::new_slice_unwrap(&[0, 64, 128, 192], NZ_2, NZ_2);

    #[test]
    fn first_channel_access() -> TestResult {
        let mut image = Image::<u8, 1>::new_vec(vec![0, 1, 2, 3], NZ_2, NZ_2)?;
        assert_eq!(image.first().dimensions(), (NZ_2, NZ_2));
        image.first_mut().make_mut_flat()[0] = 9;
        assert_eq!(image.buffer(), &[9, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn create_image_from_borrowed_channels() -> TestResult {
        let channel = LUMA8;
        let image = ImageView::try_from([channel.view()])?;
        assert_eq!(image.buffer(), &[0u8, 64, 128, 192]);
        assert_eq!(image.buffer().as_ptr(), channel.buffer().as_ptr());
        assert_eq!(image.dimensions(), (NZ_2, NZ_2));
        assert_eq!(image.len_per_channel(), 4);
        assert_eq!(image.len_per_channel_flat(), 4);
        assert_eq!(image.buffers(), [[0u8, 64, 128, 192]]);
        Ok(())
    }

    #[test]
    fn image_from_single_channel() {
        let channel = LUMA8;
        let image = ImageView::from(channel.view());
        assert_eq!(image.buffer().as_ptr(), channel.buffer().as_ptr());

        let image: Image<u8, 1> = LUMA8.into();
        assert_eq!(image.dimensions(), (NZ_2, NZ_2));
        assert_eq!(image.into_channels(), [LUMA8]);
    }

    #[test]
    fn create_image_from_borrowed_channels_with_different_dimensions() -> TestResult {
        let data = [1u8, 2, 3];
        let a = ImageChannelView::new_slice(&data[..1], NZ_1, NZ_1)?;
        let b = ImageChannelView::new_slice(&data[1..], NZ_2, NZ_1)?;
        assert_matches!(
            ImageView::try_from([a, b]),
            Err(IncompatibleImage {
                image,
                error: IncompatibleImageError::MixedImageSizes { .. },
                ..
            }) if image[1].buffer() == [2, 3]
        );
        Ok(())
    }

    #[test]
    #[should_panic = "doesn't match"]
    fn new_slice_unwrap_with_wrong_size_panics() {
        let _ = ImageView::<u8, 2>::new_slice_unwrap(&[1, 2, 3], NZ_2, NZ_1);
    }

    #[test]
    fn new_slice_borrows_planar_buffer() -> TestResult {
        let data = [1u8, 2, 3, 4];
        let mut image = ImageView::<u8, 2>::new_slice(&data, NZ_2, NZ_1)?;
        assert_eq!(image.buffers(), [[1, 2], [3, 4]]);
        assert_eq!(image.buffers()[1].as_ptr(), data[2..].as_ptr());

        let [_, second] = image.make_mut();
        second[0] = 5;
        assert_eq!(image.buffers(), [[1, 2], [5, 4]]);
        assert_eq!(data, [1, 2, 3, 4], "The borrowed buffer is copied on write");
        Ok(())
    }

    #[test]
    fn new_slice_mut_writes_into_planar_buffer() -> TestResult {
        let mut data = [[1u8, 2], [3, 4]];
        let mut image = ImageView::<[u8; 2], 1>::new_slice_mut(&mut data, NZ_2, NZ_1)?;
        image.make_mut_flat()[3] = 5;
        let clone = image.clone();
        image.make_mut()[0][0][0] = 6;
        assert_eq!(clone.buffer(), &[[1, 2], [3, 5]], "Clones copy the buffer");
        drop((image, clone));
        assert_eq!(data, [[6, 2], [3, 5]]);
        Ok(())
    }

    #[test]
    fn new_slice_with_wrong_size_fails() {
        assert_matches!(
            ImageView::<u8, 2>::new_slice(&[1, 2, 3], NZ_1, NZ_1),
            Err(InvalidLenError::BufferMismatch {
                expected: 2,
                actual: 3
            })
        );
    }

    #[test]
    fn ensure_valid_len_of_image() {
        assert_eq!(
            ImageView::<[u8; 3], 2>::ensure_valid_len(NZ_2, NZ_1, 4),
            Ok(())
        );
        assert_eq!(
            ImageView::<[u8; 3], 2>::ensure_valid_len(NZ_2, NZ_1, 12),
            Err(InvalidLenError::BufferMismatch {
                expected: 4,
                actual: 12
            })
        );
        assert_eq!(
            ImageView::<[u8; 3], 2>::ensure_valid_flat_len(NZ_2, NZ_1, 12),
            Ok(())
        );
        assert_eq!(
            ImageView::<[u8; 3], 2>::ensure_valid_flat_len(NZ_2, NZ_1, 4),
            Err(InvalidLenError::BufferMismatch {
                expected: 12,
                actual: 4
            })
        );
    }

    #[test]
    fn view_of_owned_image() -> TestResult {
        let mut image = Image::<u8, 2>::new_vec(vec![1, 2, 3, 4], NZ_2, NZ_1)?;
        let ptr = image.buffers()[1].as_ptr();
        {
            let mut view = image.view();
            assert_eq!(view.buffers()[1].as_ptr(), ptr);
            view.make_mut()[1][0] = 5;
            assert_eq!(view.buffers(), [[1, 2], [5, 4]]);
        }
        assert_eq!(image.buffers(), [[1, 2], [3, 4]]);
        {
            let mut view = image.view_mut();
            view.make_mut()[1][0] = 5;
            assert_eq!(view.buffers()[1].as_ptr(), ptr);
        }
        assert_eq!(image.buffers(), [[1, 2], [5, 4]]);
        Ok(())
    }

    #[test]
    fn view_of_view() -> TestResult {
        let mut data = [1u8, 2];
        let mut image = ImageView::<u8, 1>::new_slice_mut(&mut data, NZ_1, NZ_2)?;
        image.view_mut().view_mut().make_mut_flat()[1] = 3;
        assert_eq!(image.view().view().buffer(), &[1, 3]);
        drop(image);
        assert_eq!(data, [1, 3]);
        Ok(())
    }

    /// Owned images can be used where borrowed images are expected
    #[test]
    fn image_is_covariant() -> TestResult {
        fn shorten<'a>(image: Image<u8, 1>) -> ImageView<'a, u8, 1> {
            image
        }
        let image = Image::<u8, 1>::new_vec(vec![1], NZ_1, NZ_1)?;
        let data = [1u8];
        let borrowed = ImageView::<u8, 1>::new_slice(&data, NZ_1, NZ_1)?;
        assert_eq!(shorten(image.clone()), borrowed);
        assert_eq!(borrowed, image);
        Ok(())
    }

    #[test]
    fn miri_create_and_clear_vec_image() -> TestResult {
        let image = crate::Image::<u8, 1>::new_vec(vec![0u8, 64, 128, 192], NZ_2, NZ_2)?;
        assert_eq!(image.buffers()[0], &[0u8, 64, 128, 192]);
        assert_eq!(image.buffer(), &[0u8, 64, 128, 192]);
        Ok(())
    }

    /// Checks that the buffer was reused (e.g. transposed in place). Miri creates a new address
    /// for each use of a generic function, so the storage of a channel isn't detected and the
    /// buffer is copied instead (see `miri_to_vec_reuses_pointer`).
    fn assert_reused(buffer: *const u8, original: *const u8) {
        if !cfg!(miri) {
            assert_eq!(buffer, original, "The buffer is reused");
        }
    }

    #[test]
    fn transpose_planar_image() -> TestResult {
        let image = crate::Image::<u8, 3>::new_vec((0..12).collect(), NZ_2, NZ_2)?;
        let ptr = image.first().buffer().as_ptr();
        let interleaved_image: Image<[u8; 3], 1> = image.transpose();
        assert_eq!(
            interleaved_image.buffer(),
            &[[0u8, 4, 8], [1, 5, 9,], [2, 6, 10,], [3, 7, 11]]
        );
        assert_eq!(interleaved_image.dimensions(), (NZ_2, NZ_2));
        assert_reused(interleaved_image.buffer_flat().as_ptr(), ptr);
        Ok(())
    }

    #[test]
    fn transpose_planar_luma() -> TestResult {
        let image = crate::Image::<u8, 1>::new_vec(vec![0u8, 64, 128, 192], NZ_2, NZ_2)?;
        let ptr = image.buffer().as_ptr();
        let planar_image: Image<[u8; 1], 1> = image.transpose();
        assert_eq!(planar_image.buffer(), &[[0u8], [64u8], [128u8], [192u8]]);
        assert_reused(planar_image.buffer_flat().as_ptr(), ptr);
        Ok(())
    }

    #[test]
    fn miri_transpose_interleaved_in_place() -> TestResult {
        let image = Image::<[u8; 3], 1>::new_vec(vec![[0, 1, 2], [3, 4, 5]], NZ_2, NZ_1)?;
        let ptr = image.buffer_flat().as_ptr();
        let clone = image.clone();

        let planar = image.transpose();
        assert_eq!(planar.buffers(), [[0, 3], [1, 4], [2, 5]]);
        assert_reused(planar.first().buffer().as_ptr(), ptr);
        assert_eq!(planar.dimensions(), (NZ_2, NZ_1));
        assert_eq!(clone.buffer(), &[[0, 1, 2], [3, 4, 5]]);

        let interleaved = planar.transpose();
        assert_eq!(interleaved, clone);
        assert_reused(interleaved.buffer_flat().as_ptr(), ptr);
        Ok(())
    }

    #[test]
    fn miri_transpose_copies_shared_planar_buffer() -> TestResult {
        let image = Image::<u8, 2>::new_vec(vec![0, 1, 2, 3], NZ_2, NZ_1)?;
        let ptr = image.first().buffer().as_ptr();
        let channel = image.first().clone();

        let interleaved = image.clone().transpose();
        assert_eq!(interleaved.buffer(), &[[0, 2], [1, 3]]);
        assert_ne!(interleaved.buffer_flat().as_ptr(), ptr);
        assert_eq!(image.buffers(), [[0, 1], [2, 3]]);

        drop(image);
        // The cloned channel still shares the buffer
        let [_, second] = Image::<u8, 2>::new_vec(vec![4, 5, 6, 7], NZ_2, NZ_1)?.into_channels();
        let mixed = Image::try_from([channel.clone(), second])?.transpose();
        assert_eq!(mixed.buffer(), &[[0, 6], [1, 7]]);
        assert_eq!(channel.buffer(), &[0, 1]);
        Ok(())
    }

    #[test]
    fn miri_transpose_keeps_channel_order() -> TestResult {
        let image = Image::<u8, 3>::new_vec(vec![0, 1, 2, 3, 4, 5], NZ_2, NZ_1)?;
        let [r, g, b] = image.into_channels();
        let bgr = Image::try_from([b, g, r])?;
        assert_eq!(bgr.transpose().buffer(), &[[4, 2, 0], [5, 3, 1]]);
        Ok(())
    }

    #[test]
    fn miri_transpose_copies_borrowed_images() -> TestResult {
        let planar = Image::<u16, 2>::new_vec(vec![0, 1, 2, 3], NZ_1, NZ_2)?;
        assert_eq!(planar.view().transpose().buffer(), &[[0, 2], [1, 3]]);
        assert_eq!(planar.buffers(), [[0, 1], [2, 3]]);

        let data = [[0u16, 1], [2, 3]];
        let interleaved = ImageView::<[u16; 2], 1>::new_slice(&data, NZ_1, NZ_2)?;
        assert_eq!(interleaved.transpose().buffers(), [[0, 2], [1, 3]]);
        assert_eq!(data, [[0, 1], [2, 3]]);
        Ok(())
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Slow, the miri_transpose_* tests cover the unsafe code"
    )]
    fn transpose_large_image_in_place() -> TestResult {
        // 89 * 97 has no divisor near the tile length of 4096 bytes, so the planes are split into
        // two tiles and a tail
        let (width, height) = (
            NonZeroU32::new(89).ok_or("zero")?,
            NonZeroU32::new(97).ok_or("zero")?,
        );
        let len = 89 * 97;
        let planar = (0..3 * len)
            .map(|i| u8::try_from(i % 251))
            .collect::<Result<Vec<_>, _>>()?;
        let expected = (0..len)
            .map(|i| [planar[i], planar[len + i], planar[2 * len + i]])
            .collect::<Vec<_>>();

        let image = Image::<u8, 3>::new_vec(planar.clone(), width, height)?;
        let interleaved = image.transpose();
        assert_eq!(interleaved.buffer(), expected);
        assert_eq!(interleaved.transpose().into_vec(), planar);
        Ok(())
    }

    #[test]
    fn transpose_interleaved_luma() -> TestResult {
        let interleaved = Image::<[u8; 1], 1>::new_vec(vec![[0], [64], [128], [192]], NZ_2, NZ_2)?;
        let ptr = interleaved.buffer_flat().as_ptr();
        let planar = interleaved.transpose();
        assert_eq!(planar.buffers(), [[0u8, 64, 128, 192]]);
        assert_eq!(planar.dimensions(), (NZ_2, NZ_2));
        assert_reused(planar.buffer().as_ptr(), ptr);
        Ok(())
    }

    #[test]
    fn planar_from_borrowed_interleaved_buffer() -> TestResult {
        let data = (0..12).collect::<Vec<u8>>();
        let interleaved = ImageView::<[u8; 3], 1>::new_slice(data.as_chunks().0, NZ_2, NZ_2)?;
        let image: Image<u8, 3> = interleaved.transpose();
        assert_eq!(
            image.buffers(),
            [[0u8, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
        );
        assert_eq!(image.dimensions(), (NZ_2, NZ_2));
        Ok(())
    }

    #[test]
    fn miri_to_vec_reuses_pointer() -> TestResult {
        let raw = vec![0u8, 64, 128, 192];
        let pointer = raw[..].as_ptr();
        let image = crate::Image::<u8, 1>::new_vec(raw, NZ_2, NZ_2)?;
        let to_vec = image.into_vec();

        // Miri seems to generate clear_vec::<const u8> for each call
        // It works on native x86. Because it's only an optimization, this is good enough
        // VTable is not possible, as Image is ABI-Stable and multiple dylibs use their own allocator for Vecs
        if !cfg!(miri) {
            assert_eq!(
                to_vec[..].as_ptr(),
                pointer,
                "Should reuse the buffer if it was created by vec"
            );
        }
        Ok(())
    }

    #[test]
    fn miri_clone_from_box() -> TestResult {
        let raw = vec![0u8, 64, 128, 192];
        let image = crate::Image::<u8, 1>::new_vec(raw, NZ_2, NZ_2)?;
        let image2 = image.clone();
        let to_vec = image.into_vec();
        let to_vec2 = image2.into_vec();

        assert_ne!(
            to_vec[..].as_ptr(),
            to_vec2[..].as_ptr(),
            "Should reuse the buffer if it was created by vec"
        );
        Ok(())
    }
}
