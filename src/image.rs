use std::{
    borrow::BorrowMut,
    fmt::{self, Debug, Formatter},
    num::NonZeroU32,
    sync::Arc,
};

use crate::{
    IncompatibleImageError,
    channel::{BorrowableImageChannel, ImageChannel, calc_pixel_len_flat},
    dynamic::IncompatibleImageErrorReason,
    pixel::{PixelType, PixelTypePrimitive, RuntimePixelType},
    unwrap_usize_to_nonzero_u8,
};

pub type Image<T, const CHANNELS: usize> = ImageChannels<[ImageChannel<T>; CHANNELS]>;
pub type ImageRef<'a, T, const CHANNELS: usize> = ImageChannels<[&'a ImageChannel<T>; CHANNELS]>;
pub type ImageMut<'a, T, const CHANNELS: usize> =
    ImageChannels<[&'a mut ImageChannel<T>; CHANNELS]>;

/// Represents a image, where all channels share the same width, height. You usually want to use its typedef versions [`Image`], [`ImageRef`], [`ImageMut`] instead.
#[derive(Clone)]
#[repr(transparent)]
pub struct ImageChannels<T>(T);

impl<T: BorrowableImageChannel, const CHANNELS: usize> PartialEq for ImageChannels<[T; CHANNELS]> {
    fn eq(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| a.borrow() == b.borrow())
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
    pub fn new_vec(mut input: Vec<T>, width: NonZeroU32, height: NonZeroU32) -> Self
    where
        T: Clone,
    {
        assert_eq!(
            input.len(),
            calc_pixel_len_flat(
                width,
                height,
                const { unwrap_usize_to_nonzero_u8(CHANNELS) }
            ),
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
            let ptr = input.as_mut_ptr().cast::<T::Primitive>();
            let len = input.len() * T::ELEMENTS.get() as usize;
            let cap = input.capacity() * T::ELEMENTS.get() as usize;
            std::mem::forget(input);
            let cast_input = unsafe { Vec::from_raw_parts(ptr, len, cap) };
            Self(crate::shared_vec::create_shared_channels(
                cast_input,
                [(width, height); CHANNELS],
            ))
        }
    }

    #[must_use]
    pub fn into_channels(self) -> [ImageChannel<T>; CHANNELS] {
        self.0
    }

    #[must_use]
    pub fn into_vec(mut self) -> Vec<T>
    where
        T: Clone,
    {
        if CHANNELS == 1 {
            // For single channel, use ImageChannel::into_vec which preserves pointer reuse
            // SAFETY: When CHANNELS == 1, we know the array has exactly one element
            let ch_ptr = self.0.as_mut_ptr();
            std::mem::forget(self);
            let channel = unsafe { std::ptr::read(ch_ptr) };

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

    #[must_use]
    pub fn from_interleaved(i: &Image<[T; CHANNELS], 1>) -> Self
    where
        T: PixelTypePrimitive + Copy,
    {
        let (width, height) = i.dimensions();
        Self::from_flat_interleaved(i.buffer_flat(), (width, height))
    }

    /// # Panics
    /// Panics if the buffer size is not compatible with the width, height, and channel count.
    #[must_use]
    pub fn from_flat_interleaved(v: &[T], (width, height): (NonZeroU32, NonZeroU32)) -> Self
    where
        T: Copy,
    {
        let len = width.get() as usize * height.get() as usize;
        if CHANNELS == 1 {
            return Self::new_vec(v.to_vec(), width, height);
        }

        assert_eq!(
            v.len(),
            calc_pixel_len_flat(
                width,
                height,
                const { unwrap_usize_to_nonzero_u8(CHANNELS) }
            ),
            "Incompatible Buffer-Size"
        );

        let mut write_buf_container = vec![std::mem::MaybeUninit::<T>::uninit(); len * CHANNELS];

        let mut next_read = 0;

        let area = (width.get() * height.get()) as usize;
        let write_offsets: [_; CHANNELS] = std::array::from_fn(|i| i * area);

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

impl<T: BorrowableImageChannel + BorrowMut<ImageChannel<T::Pixel>>, const CHANNELS: usize>
    ImageChannels<[T; CHANNELS]>
{
    #[allow(clippy::missing_panics_doc)]
    pub fn make_mut(&mut self) -> [&mut [T::Pixel]; CHANNELS] {
        let mut iter = self.0.iter_mut();
        std::array::from_fn(|_| iter.next().unwrap().borrow_mut().make_mut())
    }
}
impl<T: BorrowableImageChannel, const CHANNELS: usize> ImageChannels<[T; CHANNELS]> {
    #[must_use]
    pub fn width(&self) -> NonZeroU32 {
        // All channels have the same height (validated at construction)
        // CHANNELS is always > 0
        self.0[0].borrow().width()
    }

    #[must_use]
    pub fn height(&self) -> NonZeroU32 {
        // All channels have the same height (validated at construction)
        // CHANNELS is always > 0
        self.0[0].borrow().height()
    }

    #[must_use]
    pub fn dimensions(&self) -> (NonZeroU32, NonZeroU32) {
        // All channels have the same height (validated at construction)
        // CHANNELS is always > 0
        self.0[0].borrow().dimensions()
    }

    #[must_use]
    pub fn dimensions_zeroable(&self) -> (u32, u32) {
        self.0[0].borrow().dimensions_zeroable()
    }

    /// Returns the number of pixels in each image channel
    #[must_use]
    pub fn len_per_channel(&self) -> usize {
        self.0[0].borrow().len()
    }

    #[must_use]
    pub fn len_per_channel_flat(&self) -> usize {
        self.0[0].borrow().len_flat()
    }

    #[must_use]
    pub fn buffers(&self) -> [&[T::Pixel]; CHANNELS] {
        let mut uninit = [&[] as &[T::Pixel]; CHANNELS];
        let mut i = 0;
        while i < CHANNELS {
            uninit[i] = self.0[i].borrow().buffer();
            i += 1;
        }

        uninit
    }
}

impl<T> Image<T, 1>
where
    T: PixelType,
{
    pub fn new_arc(input: Arc<[T]>, width: NonZeroU32, height: NonZeroU32) -> Self
    where
        T::Primitive: Clone,
    {
        let channel = ImageChannel::new_arc(input, width, height);
        Self([channel])
    }

    #[must_use]
    pub fn from_channel(ch: ImageChannel<T>) -> Self {
        Self([ch])
    }
}

impl<T: BorrowableImageChannel> ImageChannels<[T; 1]> {
    #[must_use]
    pub fn buffer(&self) -> &[T::Pixel] {
        self.0[0].borrow().buffer()
    }

    #[must_use]
    pub fn buffer_flat(&self) -> &[<T::Pixel as RuntimePixelType>::Primitive] {
        self.0[0].borrow().buffer_flat()
    }
}

impl<const PIXEL_ELEMENTS: usize, T: PixelTypePrimitive> Image<[T; PIXEL_ELEMENTS], 1> {
    #[must_use]
    pub fn from_planar_image<const CHANNELS: usize>(i: &Image<T, CHANNELS>) -> Self
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
            let len = width.get() as usize * height.get() as usize;
            let mut data_vec = Vec::with_capacity(len);
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

        assert_eq!(
            {
                let mut len_iter = channels.iter().map(|c| c.len());
                let first_len = len_iter.next().unwrap();

                len_iter.fold(first_len, |acc, c| {
                    assert_eq!(c, first_len, "All channels must have the same length");
                    acc + c
                })
            },
            calc_pixel_len_flat(
                width,
                height,
                const { unwrap_usize_to_nonzero_u8(CHANNELS) }
            ),
            "Incompatible Buffer-Size"
        );

        let len = width.get() as usize * height.get() as usize;
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

impl<T: BorrowableImageChannel, const CHANNELS: usize> Debug for ImageChannels<[T; CHANNELS]> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("channels", &CHANNELS)
            .field("pixel", &std::any::type_name::<T>())
            .finish()
    }
}

impl<T: BorrowableImageChannel, const CHANNELS: usize> TryFrom<[T; CHANNELS]>
    for ImageChannels<[T; CHANNELS]>
{
    type Error = IncompatibleImageError<[T; CHANNELS]>;
    fn try_from(channels: [T; CHANNELS]) -> Result<Self, Self::Error> {
        let _assert_not_empty = const { unwrap_usize_to_nonzero_u8(CHANNELS) };

        let mut iter = channels.iter().map(|x| x.borrow().dimensions());
        let a = iter
            .next()
            .expect("Checked at comptime via _assert_not_empty");
        if let Some(b) = iter.find(|x| a != *x) {
            Err(IncompatibleImageError {
                image: channels,
                reason: IncompatibleImageErrorReason::MixedImageSizes { a, b },
            })
        } else {
            Ok(Self(channels))
        }
    }
}

impl<T: BorrowableImageChannel, const CHANNELS: usize> From<ImageChannels<[T; CHANNELS]>>
    for [T; CHANNELS]
{
    fn from(value: ImageChannels<[T; CHANNELS]>) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    #[test]
    fn create_image_from_borrowed_channels() {
        let size = 2.try_into().unwrap();
        let channel = ImageChannel::<u8>::new_vec(vec![0u8, 64u8, 128u8, 192u8], size, size);
        let image = ImageRef::try_from([&channel]).unwrap();
        assert_eq!(image.buffer(), &[0u8, 64u8, 128u8, 192u8]);
        assert_eq!(image.dimensions(), (size, size));
        assert_eq!(image.len_per_channel(), 4);
        assert_eq!(image.len_per_channel_flat(), 4);
        assert_eq!(image.buffers(), [[0u8, 64u8, 128u8, 192u8]]);
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
            crate::Image::from_flat_interleaved(&[0u8, 64u8, 128u8, 192u8], (two, two));
        assert_eq!(interleaved_image.buffers(), [[0u8, 64u8, 128u8, 192u8]]);
        assert_eq!(interleaved_image.dimensions(), (two, two));
    }
    #[test]
    fn from_flat_interleaved_image() {
        let two = NonZeroU32::new(2).unwrap();
        let image: crate::Image<u8, 3> =
            Image::from_flat_interleaved((0..12).collect::<Vec<_>>().as_slice(), (two, two));
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
