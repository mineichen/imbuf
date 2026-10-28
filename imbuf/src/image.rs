use std::{
    fmt::{self, Debug, Formatter},
    num::NonZeroU32,
    sync::Arc,
};

use crate::{
    DynamicImageChannelView, IncompatibleImageError,
    channel::{ImageChannel, ImageChannelView},
    dynamic::IncompatibleImageErrorReason,
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
    /// # Panics
    /// Panics if the buffer size is not compatible with the width and height.
    #[must_use]
    pub fn new_vec_flat(mut input: Vec<T::Primitive>, width: NonZeroU32, height: NonZeroU32) -> Self
    where
        T: PixelType,
    {
        let pixel_elements = T::ELEMENTS.get() as usize;
        assert_eq!(input.len() % pixel_elements, 0, "Incompatible Buffer-Size");
        if !input.capacity().is_multiple_of(pixel_elements) {
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

    /// # Panics
    /// Panics if the buffer size is not compatible with the width and height.
    #[must_use]
    pub fn new_vec(input: Vec<T>, width: NonZeroU32, height: NonZeroU32) -> Self
    where
        T: Clone,
    {
        // Panic, if the multiplication of the dimensions overflows or the length mismatches,
        // instead of comparing against a wrapped value
        assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(
                width,
                height,
                const { unwrap_usize_to_nonzero_u8(CHANNELS) },
                input.len()
            ),
            Ok(()),
            "Incompatible Buffer-Size"
        );

        if CHANNELS == 1 {
            let channel = ImageChannel::<T>::new_vec(input, width, height);
            unsafe {
                let mut arr = std::mem::MaybeUninit::<[ImageChannel<T>; CHANNELS]>::uninit();
                std::ptr::write(arr.as_mut_ptr().cast::<ImageChannel<T>>(), channel);
                Self(arr.assume_init())
            }
        } else {
            let flat_input = crate::pixel::flat_vec(input);
            Self(crate::shared_vec::create_shared_channels(
                flat_input,
                [(width, height); CHANNELS],
            ))
        }
    }

    #[must_use]
    pub fn from_interleaved(i: &ImageView<'_, [T; CHANNELS], 1>) -> Self
    where
        T: PixelTypePrimitive + Copy,
    {
        let (width, height) = i.dimensions();
        Self::from_flat_interleaved(i.buffer_flat(), width, height)
    }

    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height, and channel count.
    #[must_use]
    pub fn from_flat_interleaved(v: &[T], width: NonZeroU32, height: NonZeroU32) -> Self
    where
        T: Copy,
    {
        if CHANNELS == 1 {
            return Self::new_vec(v.to_vec(), width, height);
        }

        // Panic, if the multiplication of the dimensions overflows or the length mismatches,
        // instead of comparing against a wrapped value
        assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(
                width,
                height,
                const { unwrap_usize_to_nonzero_u8(CHANNELS) },
                v.len()
            ),
            Ok(()),
            "Incompatible Buffer-Size"
        );

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

#[allow(clippy::len_without_is_empty)]
impl<'a, T: PixelType, const CHANNELS: usize> ImageView<'a, T, CHANNELS> {
    /// Borrows the planar `input` (all pixels of the first channel, then the second...) without
    /// copying. Mutable access (e.g. [`ImageView::make_mut`]) copies the buffers, `input` stays
    /// unchanged.
    ///
    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height and channel count.
    #[must_use]
    pub fn new_slice(input: &'a [T], width: NonZeroU32, height: NonZeroU32) -> Self {
        let mut chunks = input.chunks_exact(Self::slice_chunk_len(input.len(), width, height));
        Self(std::array::from_fn(|_| {
            let chunk = chunks.next().expect("Length is checked");
            ImageChannelView::new_slice(chunk, width, height)
        }))
    }

    /// Mutably borrows the planar `input` without copying. Mutable access
    /// (e.g. [`ImageView::make_mut`]) writes into `input`. Clones copy the buffers.
    ///
    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height and channel count.
    #[must_use]
    pub fn new_slice_mut(input: &'a mut [T], width: NonZeroU32, height: NonZeroU32) -> Self {
        let chunk_len = Self::slice_chunk_len(input.len(), width, height);
        let mut chunks = input.chunks_exact_mut(chunk_len);
        Self(std::array::from_fn(|_| {
            let chunk = chunks.next().expect("Length is checked");
            ImageChannelView::new_slice_mut(chunk, width, height)
        }))
    }

    fn slice_chunk_len(len: usize, width: NonZeroU32, height: NonZeroU32) -> usize {
        // Panic, if the multiplication of the dimensions overflows or the length mismatches,
        // instead of comparing against a wrapped value
        assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(
                width,
                height,
                const { unwrap_usize_to_nonzero_u8(CHANNELS) },
                len
            ),
            Ok(()),
            "Incompatible Buffer-Size"
        );
        // The length is checked above to be `width * height * CHANNELS`, so the division is exact
        len / CHANNELS
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
    pub fn new_arc(input: Arc<[T]>, width: NonZeroU32, height: NonZeroU32) -> Self {
        let channel = ImageChannel::new_arc(input, width, height);
        Self([channel])
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
    pub fn from_planar_image<const CHANNELS: usize>(i: &ImageView<'_, T, CHANNELS>) -> Self
    where
        T: Copy,
    {
        let (width, height) = i.dimensions();
        Self::from_planar(i.buffers(), width, height)
    }

    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height, and channel count.
    #[must_use]
    pub fn from_planar<const CHANNELS: usize>(
        channels: [&[T]; CHANNELS],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Self
    where
        T: Copy,
    {
        if CHANNELS == 1 {
            let mut data_vec = Vec::with_capacity(channels[0].len());
            for &val in channels[0] {
                data_vec.push([val; PIXEL_ELEMENTS]);
            }
            let channel = ImageChannel::new_vec(data_vec, width, height);

            return {
                let mut arr =
                    std::mem::MaybeUninit::<[ImageChannel<[T; PIXEL_ELEMENTS]>; 1]>::uninit();
                unsafe {
                    std::ptr::write(arr.as_mut_ptr().cast(), channel);
                    Self(arr.assume_init())
                }
            };
        }

        let mut len_iter = channels.iter().map(|c| c.len());
        let first_len = len_iter.next().unwrap();

        // Panic, if the sum overflows, instead of comparing against a wrapped value
        let total = len_iter
            .try_fold(first_len, |acc, len| {
                assert_eq!(len, first_len, "All channels must have the same length");
                acc.checked_add(len)
            })
            .expect("The buffer length overflows usize");

        // Panic, if the multiplication of the dimensions overflows or the length mismatches,
        // instead of comparing against a wrapped value
        assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(
                width,
                height,
                const { unwrap_usize_to_nonzero_u8(CHANNELS) },
                total
            ),
            Ok(()),
            "Incompatible Buffer-Size"
        );

        // All channels have the same length and their sum is checked above to be
        // `width * height * CHANNELS`, so each channel is `width * height` long
        let len = first_len;
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

        let image = ImageChannel::new_arc(data, width, height);
        Self([image])
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
) -> Result<(), IncompatibleImageErrorReason> {
    let Some(a) = dimensions.next() else {
        return Ok(());
    };
    match dimensions.find(|b| a != *b) {
        Some(b) => Err(IncompatibleImageErrorReason::MixedImageSizes { a, b }),
        None => Ok(()),
    }
}

impl<'a, T: PixelType, const CHANNELS: usize> TryFrom<[ImageChannelView<'a, T>; CHANNELS]>
    for ImageView<'a, T, CHANNELS>
{
    type Error = IncompatibleImageError<[ImageChannelView<'a, T>; CHANNELS]>;
    fn try_from(channels: [ImageChannelView<'a, T>; CHANNELS]) -> Result<Self, Self::Error> {
        let _assert_not_empty = const { unwrap_usize_to_nonzero_u8(CHANNELS) };

        match check_dimensions(channels.iter().map(ImageChannelView::dimensions)) {
            Ok(()) => Ok(Self(channels)),
            Err(reason) => Err(IncompatibleImageError {
                image: channels,
                reason,
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
    use std::num::NonZeroU32;

    use super::*;

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

    #[test]
    fn first_channel_access() {
        let mut image = Image::<u8, 1>::new_vec(vec![0, 1, 2, 3], NZ_2, NZ_2);
        assert_eq!(image.first().dimensions(), (NZ_2, NZ_2));
        image.first_mut().make_mut_flat()[0] = 9;
        assert_eq!(image.buffer(), &[9, 1, 2, 3]);
    }

    #[test]
    fn create_image_from_borrowed_channels() {
        let size = 2.try_into().unwrap();
        let channel = ImageChannel::<u8>::new_vec(vec![0u8, 64u8, 128u8, 192u8], size, size);
        let image = ImageView::try_from([channel.view()]).unwrap();
        assert_eq!(image.buffer(), &[0u8, 64u8, 128u8, 192u8]);
        assert_eq!(image.buffer().as_ptr(), channel.buffer().as_ptr());
        assert_eq!(image.dimensions(), (size, size));
        assert_eq!(image.len_per_channel(), 4);
        assert_eq!(image.len_per_channel_flat(), 4);
        assert_eq!(image.buffers(), [[0u8, 64u8, 128u8, 192u8]]);
    }

    #[test]
    fn create_image_from_borrowed_channels_with_different_dimensions() {
        let data = [1u8, 2, 3];
        let a = ImageChannelView::new_slice(&data[..1], NZ_1, NZ_1);
        let b = ImageChannelView::new_slice(&data[1..], NZ_2, NZ_1);
        let err = ImageView::try_from([a, b]).unwrap_err();
        assert_eq!(err.image[1].buffer(), &[2, 3]);
    }

    #[test]
    fn new_slice_borrows_planar_buffer() {
        let two = NonZeroU32::new(2).unwrap();
        let data = [1u8, 2, 3, 4];
        let mut image = ImageView::<u8, 2>::new_slice(&data, two, NonZeroU32::MIN);
        assert_eq!(image.buffers(), [[1, 2], [3, 4]]);
        assert_eq!(image.buffers()[1].as_ptr(), data[2..].as_ptr());

        let [_, second] = image.make_mut();
        second[0] = 5;
        assert_eq!(image.buffers(), [[1, 2], [5, 4]]);
        assert_eq!(data, [1, 2, 3, 4], "The borrowed buffer is copied on write");
    }

    #[test]
    fn new_slice_mut_writes_into_planar_buffer() {
        let two = NonZeroU32::new(2).unwrap();
        let mut data = [[1u8, 2], [3, 4]];
        let mut image = ImageView::<[u8; 2], 1>::new_slice_mut(&mut data, two, NonZeroU32::MIN);
        image.make_mut_flat()[3] = 5;
        let clone = image.clone();
        image.make_mut()[0][0][0] = 6;
        assert_eq!(clone.buffer(), &[[1, 2], [3, 5]], "Clones copy the buffer");
        drop((image, clone));
        assert_eq!(data, [[6, 2], [3, 5]]);
    }

    #[test]
    #[should_panic(expected = "Incompatible Buffer-Size")]
    fn new_slice_with_wrong_size_panics() {
        let _ = ImageView::<u8, 2>::new_slice(&[1, 2, 3], NonZeroU32::MIN, NonZeroU32::MIN);
    }

    #[test]
    fn view_of_owned_image() {
        let two = NonZeroU32::new(2).unwrap();
        let mut image = Image::<u8, 2>::new_vec(vec![1, 2, 3, 4], two, NonZeroU32::MIN);
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
    }

    #[test]
    fn view_of_view() {
        let mut data = [1u8, 2];
        let mut image = ImageView::<u8, 1>::new_slice_mut(
            &mut data,
            NonZeroU32::MIN,
            NonZeroU32::new(2).unwrap(),
        );
        image.view_mut().view_mut().make_mut_flat()[1] = 3;
        assert_eq!(image.view().view().buffer(), &[1, 3]);
        drop(image);
        assert_eq!(data, [1, 3]);
    }

    /// Owned images can be used where borrowed images are expected
    #[test]
    fn image_is_covariant() {
        fn shorten<'a>(image: Image<u8, 1>) -> ImageView<'a, u8, 1> {
            image
        }
        let image = Image::<u8, 1>::new_vec(vec![1], NonZeroU32::MIN, NonZeroU32::MIN);
        let data = [1u8];
        let borrowed = ImageView::<u8, 1>::new_slice(&data, NonZeroU32::MIN, NonZeroU32::MIN);
        assert_eq!(shorten(image.clone()), borrowed);
        assert_eq!(borrowed, image);
    }

    #[test]
    fn miri_create_and_clear_vec_image() {
        let size = 2.try_into().unwrap();
        let image = crate::Image::<u8, 1>::new_vec(vec![0u8, 64u8, 128u8, 192u8], size, size);
        assert_eq!(image.buffers()[0], &[0u8, 64u8, 128u8, 192u8]);
        assert_eq!(image.buffer(), &[0u8, 64u8, 128u8, 192u8]);
    }

    #[test]
    fn from_planar_image() {
        let two = NonZeroU32::new(2).unwrap();
        let image = crate::Image::<u8, 3>::new_vec((0..12).collect(), two, two);
        let interleaved_image = crate::Image::<[u8; 3], 1>::from_planar_image(&image);
        assert_eq!(
            interleaved_image.buffer(),
            &[[0u8, 4, 8], [1, 5, 9,], [2, 6, 10,], [3, 7, 11]]
        );
        assert_eq!(interleaved_image.dimensions(), (two, two));
    }

    #[test]
    fn luma_from_planar() {
        let two = NonZeroU32::new(2).unwrap();
        let image = crate::Image::<u8, 1>::new_vec(vec![0u8, 64u8, 128u8, 192u8], two, two);
        let planar_image = Image::<[u8; 1], 1>::from_planar_image(&image);
        assert_eq!(planar_image.buffer(), &[[0u8], [64u8], [128u8], [192u8]]);
    }

    #[test]
    fn luma_from_interleaved() {
        let two = NonZeroU32::new(2).unwrap();
        let interleaved_image =
            crate::Image::from_flat_interleaved(&[0u8, 64u8, 128u8, 192u8], two, two);
        assert_eq!(interleaved_image.buffers(), [[0u8, 64u8, 128u8, 192u8]]);
        assert_eq!(interleaved_image.dimensions(), (two, two));
    }
    #[test]
    fn from_flat_interleaved_image() {
        let two = NonZeroU32::new(2).unwrap();
        let image: crate::Image<u8, 3> =
            Image::from_flat_interleaved((0..12).collect::<Vec<_>>().as_slice(), two, two);
        assert_eq!(
            image.buffers(),
            [[0u8, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
        );
        assert_eq!(image.dimensions(), (two, two));
    }

    #[test]
    fn miri_to_vec_reuses_pointer() {
        let raw = vec![0u8, 64u8, 128u8, 192u8];
        let pointer = raw[..].as_ptr();
        let size = 2.try_into().unwrap();
        let image = crate::Image::<u8, 1>::new_vec(raw, size, size);
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
    }

    #[test]
    fn miri_clone_from_box() {
        let raw = vec![0u8, 64u8, 128u8, 192u8];
        let size = 2.try_into().unwrap();
        let image = crate::Image::<u8, 1>::new_vec(raw, size, size);
        let image2 = image.clone();
        let to_vec = image.into_vec();
        let to_vec2 = image2.into_vec();

        assert_ne!(
            to_vec[..].as_ptr(),
            to_vec2[..].as_ptr(),
            "Should reuse the buffer if it was created by vec"
        );
    }

    #[test]
    fn create_interleaved_from_flat_vec() {
        let size = 2.try_into().unwrap();
        let image: Image<[u8; 3], 1> = Image::new_vec(
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            size,
            size,
        );
        let buffer = image.buffer_flat().to_vec();
        assert_eq!(buffer.len(), 12);
        let image_from_flat: Image<[u8; 3], 1> = Image::new_vec_flat(buffer, size, size);

        assert_eq!(image, image_from_flat);
    }
}
