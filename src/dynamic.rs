use std::{
    fmt::Debug,
    mem::MaybeUninit,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
};

use crate::{Image, ImageChannel, ImageMut, ImageRef, PixelType, pixel::DynamicSize};

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
            channels: std::iter::once(first).chain(rest).collect(),
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

    /// Checkes, if all channels have the same width, height, pixel_elements and pixel_kind
    pub fn is_uniform(&self) -> bool {
        let (first, mut rest) = self.first_with_rest_ref_mapped(|x| {
            (
                match x {
                    DynamicImageChannel::U8(_) => 0,
                    DynamicImageChannel::U16(_) => 1,
                    DynamicImageChannel::F32(_) => 2,
                },
                x.pixel_elements(),
                x.dimensions(),
            )
        });
        rest.all(|x| x == first)
    }

    pub fn first_with_rest_ref_mapped<'a, T: 'a, TF>(
        &'a self,
        mapper: TF,
    ) -> (
        T,
        std::iter::Map<std::slice::Iter<'a, DynamicImageChannel>, TF>,
    )
    where
        TF: FnMut(&'a DynamicImageChannel) -> T,
    {
        let mut iter = self.iter().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    pub fn first_with_rest_mut_mapped<'a, T: 'a, TF>(
        &'a mut self,
        mapper: TF,
    ) -> (
        T,
        std::iter::Map<std::slice::IterMut<'a, DynamicImageChannel>, TF>,
    )
    where
        TF: FnMut(&'a mut DynamicImageChannel) -> T,
    {
        let mut iter = self.iter_mut().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    pub fn first_with_rest_mapped<T, TF>(
        self,
        mapper: TF,
    ) -> (
        T,
        std::iter::Map<std::vec::IntoIter<DynamicImageChannel>, TF>,
    )
    where
        TF: FnMut(DynamicImageChannel) -> T,
    {
        let mut iter = self.into_iter().map(mapper);
        let first = iter.next().expect("Always has a image");
        (first, iter)
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.channels.len()).unwrap()
    }
}

// Deref slice only, to make sure, noone can create a DynamicImage with a empty Vec
impl std::ops::Deref for DynamicImage {
    type Target = [DynamicImageChannel];

    fn deref(&self) -> &Self::Target {
        &self.channels
    }
}

impl IntoIterator for DynamicImage {
    type Item = DynamicImageChannel;
    type IntoIter = std::vec::IntoIter<DynamicImageChannel>;

    fn into_iter(self) -> Self::IntoIter {
        self.channels.into_iter()
    }
}

impl std::ops::DerefMut for DynamicImage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.channels
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynamicImageChannel {
    U8(ImageChannel<DynamicSize<u8>>),
    U16(ImageChannel<DynamicSize<u16>>),
    F32(ImageChannel<DynamicSize<f32>>),
}

impl DynamicImageChannel {
    #[must_use]
    pub fn pixel_elements(&self) -> NonZeroU8 {
        match self {
            DynamicImageChannel::U8(x) => x.pixel_elements(),
            DynamicImageChannel::U16(x) => x.pixel_elements(),
            DynamicImageChannel::F32(x) => x.pixel_elements(),
        }
    }

    #[must_use]
    pub fn dimensions(&self) -> (NonZeroU32, NonZeroU32) {
        match self {
            DynamicImageChannel::U8(x) => x.dimensions(),
            DynamicImageChannel::U16(x) => x.dimensions(),
            DynamicImageChannel::F32(x) => x.dimensions(),
        }
    }

    #[must_use]
    pub fn width(&self) -> NonZeroU32 {
        match self {
            DynamicImageChannel::U8(x) => x.width(),
            DynamicImageChannel::U16(x) => x.width(),
            DynamicImageChannel::F32(x) => x.width(),
        }
    }

    #[must_use]
    pub fn height(&self) -> NonZeroU32 {
        match self {
            DynamicImageChannel::U8(x) => x.height(),
            DynamicImageChannel::U16(x) => x.height(),
            DynamicImageChannel::F32(x) => x.height(),
        }
    }
}

impl<TPixel: PixelType + Send + Sync + Clone, const CHANNELS: usize> From<Image<TPixel, CHANNELS>>
    for DynamicImage
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

impl<T: PixelType, const CHANNELS: usize> TryFrom<DynamicImage> for Image<T, CHANNELS> {
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
                        Err(_e) => Err(IncompatibleImageErrorReason::Comptime {
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
            };
            MaybeUninit::uninit()
        });
        result
            .and_then(|_| {
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
{
    type Error = IncompatibleImageError<&'a mut DynamicImage>;

    fn try_from(value: &'a mut DynamicImage) -> Result<Self, Self::Error> {
        let mut result = Ok(());
        let mut iter = value.channels.iter_mut();
        let channels = std::array::from_fn(|i| {
            if result.is_ok() {
                result = if let Some(next) = iter.next() {
                    match <&mut ImageChannel<T>>::try_from(next) {
                        Ok(typed) => {
                            return MaybeUninit::new(typed);
                        }
                        Err(_e) => Err(IncompatibleImageErrorReason::Comptime {
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
            };
            MaybeUninit::uninit()
        });
        result
            .and_then(|_| {
                <ImageMut<'a, T, CHANNELS>>::try_from(channels.map(|x| unsafe { x.assume_init() }))
                    .map_err(|x: IncompatibleImageError<[&mut ImageChannel<T>; CHANNELS]>| x.reason)
            })
            .map_err(|reason| IncompatibleImageError {
                image: value,
                reason,
            })
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
        let (first, mut rest) = dynamic.first_with_rest_ref_mapped(|x| match x {
            DynamicImageChannel::U8(image_channel) => image_channel.buffer_flat_bytes(),
            DynamicImageChannel::U16(image_channel) => image_channel.buffer_flat_bytes(),
            DynamicImageChannel::F32(image_channel) => image_channel.buffer_flat_bytes(),
        });
        assert_eq!(1, first[0]);
        assert_eq!(2, rest.next().unwrap()[0]);
        assert_eq!(3, rest.next().unwrap()[0]);
    }

    #[test]
    fn first_mapped_returns_mut() {
        let ch = Image::<u8, 3>::new_vec(vec![1, 2, 3], NonZeroU32::MIN, NonZeroU32::MIN);
        let mut dynamic = DynamicImage::from(ch);
        let (first, mut rest) = dynamic.first_with_rest_mut_mapped(|x| match x {
            DynamicImageChannel::U8(image_channel) => image_channel.buffer_flat_bytes(),
            DynamicImageChannel::U16(image_channel) => image_channel.buffer_flat_bytes(),
            DynamicImageChannel::F32(image_channel) => image_channel.buffer_flat_bytes(),
        });
        assert_eq!(1, first[0]);
        assert_eq!(2, rest.next().unwrap()[0]);
        assert_eq!(3, rest.next().unwrap()[0]);
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
        // assert_eq!(
        //     vec![(const { NonZeroU8::new(3).unwrap() }, DynamicPixelKind::U8)],
        //     dynamic.channel_infos().collect::<Vec<_>>()
        // );
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
