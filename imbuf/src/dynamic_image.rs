use std::{
    iter::{self, once},
    num::{NonZeroU8, NonZeroUsize},
    ops::Deref,
    slice, vec,
};

use bytemuck::Pod;

#[cfg(doc)]
use crate::PixelLayout;
use crate::{
    DynamicImageChannelView, ImageChannelView, ImageView, IncompatibleImage, PixelType,
    PixelTypePrimitive,
    error::{IncompatibleChannelCount, IncompatibleImageError},
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

    /// Mutable access to the buffer of a single channel image without copying it, see
    /// [`DynamicImageChannelView::buffer_mut`]. For images with multiple channels, use
    /// [`DynamicImageView::try_cast_mut`] and [`ImageView::buffers_mut`].
    ///
    /// Fails, if the image has more than one channel or the layout of `T` doesn't match
    pub fn buffer_mut<T: PixelType + Pod>(
        &mut self,
    ) -> Result<Option<&mut [T]>, IncompatibleImageError> {
        self.ensure_single_channel()?;
        Ok(self.channels[0].buffer_mut()?)
    }

    /// Flat version of [`DynamicImageView::buffer_mut`], see
    /// [`DynamicImageChannelView::buffer_flat_mut`]
    ///
    /// Fails, if the image has more than one channel or the size or the alignment of `T` don't
    /// match
    pub fn buffer_flat_mut<T: PixelTypePrimitive + Pod>(
        &mut self,
    ) -> Result<Option<&mut [T]>, IncompatibleImageError> {
        self.ensure_single_channel()?;
        Ok(self.channels[0].buffer_flat_mut()?)
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
    use crate::{
        DynamicImageChannel, Image, ImageChannel, ImageView, IncompatibleChannelError, PixelLayout,
    };

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    const NZ_U8_3: NonZeroU8 = NonZeroU8::new(3).unwrap();
    const RGB8: Image<[u8; 3], 1> = Image::new_slice_unwrap(&[[1, 2, 3]], NZ_1, NZ_1);
    const RGB16: Image<[u16; 3], 1> = Image::new_slice_unwrap(&[[1, 2, 3]], NZ_1, NZ_1);
    const PLANAR_RGB8: Image<u8, 3> = Image::new_slice_unwrap(&[1, 2, 3], NZ_1, NZ_1);
    const LUMA8: Image<u8, 1> = Image::new_slice_unwrap(&[1], NZ_1, NZ_1);

    #[test]
    fn ensure_single_channel() -> TestResult {
        let single = DynamicImage::from(RGB8);
        assert_eq!(single.ensure_single_channel(), Ok(()));

        let planar = DynamicImage::from(PLANAR_RGB8);
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
    fn image_with_other_dimensions_is_not_uniform() {
        let ch1 = const { ImageChannel::new_slice_unwrap(&[1u8, 1], NZ_1, NZ_2) };
        let ch2 = const { ImageChannel::new_slice_unwrap(&[1u8], NZ_1, NZ_1) };
        let dynamic = DynamicImage::from_channels(ch1.into(), [ch2.into()]);
        assert!(!dynamic.is_uniform());
    }

    #[test]
    fn image_with_other_primitive_size_is_not_uniform() {
        let ch1 = const { ImageChannel::new_slice_unwrap(&[1u8], NZ_1, NZ_1) };
        let ch2 = const { ImageChannel::new_slice_unwrap(&[1u16], NZ_1, NZ_1) };
        let dynamic = DynamicImage::from_channels(ch1.into(), [ch2.into()]);
        assert!(!dynamic.is_uniform());
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
        let dynamic = DynamicImage::from(PLANAR_RGB8);
        let (first, mut rest) = dynamic.mapped_channels(DynamicImageChannel::buffer_flat_bytes);
        assert_eq!(1, first[0]);
        assert_eq!(2, rest.next().ok_or("missing channel")?[0]);
        assert_eq!(3, rest.next().ok_or("missing channel")?[0]);
        Ok(())
    }

    #[test]
    fn first_mapped_returns_mut() -> TestResult {
        let mut dynamic = DynamicImage::from(PLANAR_RGB8);
        let (first, mut rest) = dynamic.mapped_channels_mut(|x| x.make_mut_flat::<u8>());
        first?[0] = 4;
        assert_eq!(2, rest.next().ok_or("missing channel")??[0]);
        assert_eq!(3, rest.next().ok_or("missing channel")??[0]);
        assert_eq!(dynamic.first().buffer_flat::<u8>(), Ok(&[4][..]));
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
    fn borrow_from_different_size_mut_dynamic_image() {
        let image = const { Image::<u8, 2>::new_slice_unwrap(&[0, 1], NZ_1, NZ_1) };
        let mut dynamic = DynamicImage::from(image);
        dynamic[1] = const { ImageChannel::new_slice_unwrap(&[0u8, 1], NZ_1, NZ_2) }.into();

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
        let dynamic = DynamicImage::from(RGB8);
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
        let dynamic = DynamicImage::from(LUMA8);
        assert_eq!(1, dynamic.channels.len());
        let luma_back: Image<u8, 1> = dynamic.try_into()?;
        assert_eq!(luma_back.into_vec(), vec![1]);
        Ok(())
    }

    #[test]
    fn create_from_luma_rgb16_interleaved() -> TestResult {
        let dynamic = DynamicImage::from(RGB16);
        assert_eq!(1, dynamic.channels.len());
        let luma_back: Image<[u16; 3], 1> = dynamic.try_into()?;
        assert_eq!(luma_back.into_vec(), vec![[1u16, 2, 3]]);
        Ok(())
    }

    #[test]
    fn create_from_rgb8_interleaved() -> TestResult {
        let dynamic = DynamicImage::from(RGB8);
        assert_eq!(1, dynamic.channels.len());
        assert_eq!(1, dynamic[0].primitive_size().get());
        assert_eq!(3, dynamic[0].pixel_elements().get());
        assert_eq!(Ok([[1u8, 2, 3]].as_slice()), dynamic[0].buffer::<[u8; 3]>());
        let rgb_back: Image<[u8; 3], 1> = dynamic.try_into()?;
        assert_eq!(rgb_back.into_vec(), vec![[1u8, 2, 3]]);
        Ok(())
    }
    #[test]
    fn create_from_luma_rgb8_planar() -> TestResult {
        let dynamic = DynamicImage::from(PLANAR_RGB8);
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
    fn create_from_incompatible_image() {
        let dynamic = DynamicImage::from(LUMA8);
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
                            primitive_size: NonZeroUsize::MIN,
                            primitive_alignment: NonZeroUsize::MIN
                        },
                        required: Some(PixelLayout {
                            elements: NonZeroU8::MIN,
                            primitive_size: second_primitive_size,
                            primitive_alignment: second_primitive_alignment
                        }),
                        target_pixel_type: "u16",
                        ..
                    }),
                ..
            }) if image == DynamicImage::from(LUMA8)
                && second_primitive_size.get() == 2
                && second_primitive_alignment.get() == 2
        );
    }

    #[test]
    fn create_from_image_with_other_pixel_elements() {
        let dynamic = DynamicImage::from(RGB8);
        assert_matches!(
            ImageView::<u8, 1>::try_from(&dynamic),
            Err(IncompatibleImageError::Channel(IncompatibleChannelError {
                actual: PixelLayout {
                    elements: NZ_U8_3,
                    primitive_size: first_primitive_size,
                    primitive_alignment: NonZeroUsize::MIN
                },
                required: Some(PixelLayout {
                    elements: NonZeroU8::MIN,
                    primitive_size: NonZeroUsize::MIN,
                            primitive_alignment: NonZeroUsize::MIN
                }),
                target_pixel_type: "u8",
            })) if first_primitive_size.get() == 1
        );
    }
}
