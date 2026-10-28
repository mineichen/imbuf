use std::{
    fmt::{self, Debug, Formatter},
    num::{NonZeroU8, NonZeroU32},
    sync::Arc,
};

use crate::{
    DynamicImageChannelView, IncompatibleImage, InvalidLenError,
    channel::{ImageChannel, ImageChannelView},
    error::IncompatibleImageError,
    pixel::{PixelType, PixelTypePrimitive},
    unwrap_usize_to_nonzero_u8,
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

#[allow(clippy::len_without_is_empty)]
impl<const CHANNELS: usize, T: PixelType> Image<T, CHANNELS> {
    pub fn new_vec_flat(
        mut input: Vec<T::Primitive>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError>
    where
        T: PixelType,
    {
        Self::ensure_valid_flat_len(width, height, input.len())?;
        if CHANNELS > 1 {
            // The channels share the planar buffer, so it doesn't have to be cast to `T`
            // Safety: The length is checked above
            let channels = unsafe {
                crate::shared_vec::create_shared_channels_unchecked(
                    input,
                    [(width, height); CHANNELS],
                )
            };
            return Ok(Self(channels));
        }
        let pixel_elements = T::ELEMENTS.get() as usize;
        if input.capacity() % pixel_elements != 0 {
            // Vec<T> cannot represent the allocation. Boxed slices have an exact capacity.
            input = input.into_boxed_slice().into_vec();
        }

        let cap = input.capacity() / pixel_elements;
        let len = input.len() / pixel_elements;
        let ptr = input.as_mut_ptr().cast::<T>();
        std::mem::forget(input);
        let cast_input = unsafe { Vec::from_raw_parts(ptr, len, cap) };

        Self::new_vec(cast_input, width, height)
    }

    pub fn new_vec(
        input: Vec<T>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError>
    where
        T: Clone,
    {
        Self::ensure_valid_len(width, height, input.len())?;

        if CHANNELS == 1 {
            let channel = ImageChannel::<T>::new_vec(input, width, height)?;
            unsafe {
                let mut arr = std::mem::MaybeUninit::<[ImageChannel<T>; CHANNELS]>::uninit();
                std::ptr::write(arr.as_mut_ptr().cast::<ImageChannel<T>>(), channel);
                Ok(Self(arr.assume_init()))
            }
        } else {
            let flat_input = crate::pixel::flat_vec(input);
            // Safety: The length is checked above to be `width * height * CHANNELS`, so the flat
            // length is `width * height * T::ELEMENTS * CHANNELS`, which fits `usize`, as the
            // buffer exists
            let channels = unsafe {
                crate::shared_vec::create_shared_channels_unchecked(
                    flat_input,
                    [(width, height); CHANNELS],
                )
            };
            Ok(Self(channels))
        }
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn from_interleaved(i: &ImageView<'_, [T; CHANNELS], 1>) -> Self
    where
        T: PixelTypePrimitive + Copy,
    {
        let (width, height) = i.dimensions();
        Self::from_flat_interleaved(i.buffer_flat(), width, height)
            .expect("The buffer of an image matches its dimensions")
    }

    pub fn from_flat_interleaved(
        v: &[T],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError>
    where
        T: Copy,
    {
        if CHANNELS == 1 {
            return Self::new_vec(v.to_vec(), width, height);
        }

        Self::ensure_valid_len(width, height, v.len())?;

        // Todo: Change interleaved to planar should be done inline instead of new vec
        let mut write_buf_container = vec![std::mem::MaybeUninit::<T>::uninit(); v.len()];

        let mut next_read = 0;

        // The buffer length is checked above to be `width * height * CHANNELS`, so the division
        // is exact and `i * len` cannot overflow
        let len = v.len() / CHANNELS;
        let write_offsets: [_; CHANNELS] = std::array::from_fn(|i| i * len);

        for pixel in 0..len {
            for (i, write_offset) in write_offsets.iter().enumerate() {
                unsafe {
                    write_buf_container
                        .get_unchecked_mut(pixel + write_offset)
                        .write(*v.get_unchecked(next_read + i));
                }
            }
            next_read += CHANNELS;
        }
        let x = unsafe {
            std::mem::transmute::<Vec<std::mem::MaybeUninit<T>>, Vec<T>>(write_buf_container)
        };
        Image::<T, CHANNELS>::new_vec(x, width, height)
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
    #[allow(clippy::missing_panics_doc)]
    pub fn new_slice(
        input: &'a [T],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        let len = input.len();
        Self::ensure_valid_len(width, height, len)?;
        let mut chunks = input.chunks_exact(len / CHANNELS);
        Ok(Self(std::array::from_fn(|_| {
            let chunk = chunks.next().expect("Length is checked");
            ImageChannelView::new_slice(chunk, width, height).expect("Length is checked")
        })))
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

impl<T> Image<T, 1>
where
    T: PixelType,
{
    // Todo: If shared_vec is extendedto support various owned slices, it could be implemented on all channel sizes too
    pub fn new_arc(
        input: Arc<[T]>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        let channel = ImageChannel::new_arc(input, width, height)?;
        Ok(Self([channel]))
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
}

impl<const PIXEL_ELEMENTS: usize, T: PixelTypePrimitive> Image<[T; PIXEL_ELEMENTS], 1> {
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn from_planar_image<const CHANNELS: usize>(i: &ImageView<'_, T, CHANNELS>) -> Self
    where
        T: Copy,
    {
        let (width, height) = i.dimensions();
        Self::from_planar(i.buffers(), width, height)
            .expect("The buffers of an image match its dimensions")
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn from_planar<const CHANNELS: usize>(
        channels: [&[T]; CHANNELS],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError>
    where
        T: Copy,
    {
        for channel in channels {
            DynamicImageChannelView::ensure_valid_len(width, height, channel.len())?;
        }

        if CHANNELS == 1 {
            let mut data_vec = Vec::with_capacity(channels[0].len());
            for &val in channels[0] {
                data_vec.push([val; PIXEL_ELEMENTS]);
            }
            let channel = ImageChannel::new_vec(data_vec, width, height)?;

            return Ok({
                let mut arr =
                    std::mem::MaybeUninit::<[ImageChannel<[T; PIXEL_ELEMENTS]>; 1]>::uninit();
                unsafe {
                    std::ptr::write(arr.as_mut_ptr().cast(), channel);
                    Self(arr.assume_init())
                }
            });
        }

        // Each channel is checked above to be `width * height` long
        let len = channels[0].len();
        let mut channel_iters = channels.map(|c| c.iter());

        let mut data_vec = vec![std::mem::MaybeUninit::<[T; PIXEL_ELEMENTS]>::uninit(); len];
        for dst in &mut data_vec {
            let mut value = [std::mem::MaybeUninit::<T>::uninit(); PIXEL_ELEMENTS];

            for (src, dst) in channel_iters
                .iter_mut()
                .map(|c| {
                    c.next()
                        .expect("Channels are checked above to have the same length")
                })
                .zip(value.iter_mut())
            {
                dst.write(*src);
            }

            dst.write(value.map(|x| unsafe { x.assume_init() }));
        }
        let data_vec_init: Vec<[T; PIXEL_ELEMENTS]> = unsafe { std::mem::transmute(data_vec) };
        let data = std::sync::Arc::from(data_vec_init);

        let image = ImageChannel::new_arc(data, width, height)?;
        Ok(Self([image]))
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

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::num::NonZeroU32;

    use testresult::TestResult;

    use super::*;

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

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
        let channel = ImageChannel::<u8>::new_vec(vec![0u8, 64, 128, 192], NZ_2, NZ_2)?;
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
    fn new_vec_flat_reports_len_in_primitives() {
        // 3 primitives are a whole pixel, but 2 pixels are required
        assert_matches!(
            Image::<[u8; 3], 1>::new_vec_flat(vec![1, 2, 3], NZ_2, NZ_1),
            Err(InvalidLenError::BufferMismatch {
                expected: 6,
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

    #[test]
    fn from_planar_image() -> TestResult {
        let image = crate::Image::<u8, 3>::new_vec((0..12).collect(), NZ_2, NZ_2)?;
        let interleaved_image = crate::Image::<[u8; 3], 1>::from_planar_image(&image);
        assert_eq!(
            interleaved_image.buffer(),
            &[[0u8, 4, 8], [1, 5, 9,], [2, 6, 10,], [3, 7, 11]]
        );
        assert_eq!(interleaved_image.dimensions(), (NZ_2, NZ_2));
        Ok(())
    }

    #[test]
    fn luma_from_planar() -> TestResult {
        let image = crate::Image::<u8, 1>::new_vec(vec![0u8, 64, 128, 192], NZ_2, NZ_2)?;
        let planar_image = Image::<[u8; 1], 1>::from_planar_image(&image);
        assert_eq!(planar_image.buffer(), &[[0u8], [64u8], [128u8], [192u8]]);
        Ok(())
    }

    #[test]
    fn luma_from_interleaved() -> TestResult {
        let interleaved_image =
            crate::Image::from_flat_interleaved(&[0u8, 64, 128, 192], NZ_2, NZ_2)?;
        assert_eq!(interleaved_image.buffers(), [[0u8, 64, 128, 192]]);
        assert_eq!(interleaved_image.dimensions(), (NZ_2, NZ_2));
        Ok(())
    }
    #[test]
    fn from_flat_interleaved_image() -> TestResult {
        let image: crate::Image<u8, 3> =
            Image::from_flat_interleaved((0..12).collect::<Vec<_>>().as_slice(), NZ_2, NZ_2)?;
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

    #[test]
    fn create_interleaved_from_flat_vec() -> TestResult {
        let image: Image<[u8; 3], 1> = Image::new_vec(
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            NZ_2,
            NZ_2,
        )?;
        let buffer = image.buffer_flat().to_vec();
        assert_eq!(buffer.len(), 12);
        let image_from_flat: Image<[u8; 3], 1> = Image::new_vec_flat(buffer, NZ_2, NZ_2)?;

        assert_eq!(image, image_from_flat);
        Ok(())
    }
}
