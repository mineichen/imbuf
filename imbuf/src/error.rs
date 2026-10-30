use std::{
    error::Error,
    fmt::Display,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
};

use crate::{PixelType, PixelTypePrimitive};

/// Returned by conversions of owned images, which hand back the input in `image` if it is
/// incompatible: Converting a [`DynamicImageView`](crate::DynamicImageView) into
/// `Result<ImageView, IncompatibleImage<DynamicImageView>>` with `From`/`Into` (`TryFrom` only
/// returns the [`IncompatibleImageError`]) and `TryFrom<[ImageChannelView; N]> for ImageView`.
/// As the input is returned, it doesn't
/// implement `Error` (but `Display`). Use [`IncompatibleImage::into_error`] (e.g. in `map_err`)
/// to propagate the cause, or destructure it to get both the image and the cause.
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub struct IncompatibleImage<TInput> {
    pub image: TInput,
    pub error: IncompatibleImageError,
}

impl<TInput> IncompatibleImage<TInput> {
    /// Drops the image and returns the cause
    #[must_use]
    pub fn into_error(self) -> impl Error + Send + Sync + 'static {
        self.error
    }
}

/// Prints the cause only. This makes `?` work in contexts which accept any `Display`
/// (like `testresult::TestResult`), without needing [`IncompatibleImage::into_error`].
impl<TInput> Display for IncompatibleImage<TInput> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

/// Returned by converting a [`DynamicImageChannelView`](crate::DynamicImageChannelView) into
/// `Result<ImageChannelView, IncompatibleChannel<DynamicImageChannelView>>` with `From`/`Into`
/// (`TryFrom` only returns the [`IncompatibleChannelError`]), which hands back the input in
/// `channel` if it is incompatible. As the input is returned, it
/// doesn't implement `Error` (but `Display`). Use [`IncompatibleChannel::into_error`] (e.g. in
/// `map_err`) to propagate the cause, or destructure it to get both the channel and the cause.
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub struct IncompatibleChannel<TInput> {
    pub channel: TInput,
    pub error: IncompatibleChannelError,
}

impl<TInput> IncompatibleChannel<TInput> {
    /// Drops the channel and returns the cause
    #[must_use]
    pub fn into_error(self) -> impl Error + Send + Sync + 'static {
        self.error
    }
}

/// Prints the cause only. This makes `?` work in contexts which accept any `Display`
/// (like `testresult::TestResult`), without needing [`IncompatibleChannel::into_error`].
impl<TInput> Display for IncompatibleChannel<TInput> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

/// Error of [`DynamicImageChannelView::ensure_valid_flat_len`](crate::DynamicImageChannelView::ensure_valid_flat_len)
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InvalidLenError {
    #[error(transparent)]
    Overflow(#[from] OverflowBufferLenError),
    #[error("The buffer length {actual} doesn't match the dimensions, which require {expected}")]
    BufferMismatch { expected: usize, actual: usize },
}

/// Reason, why a [`DynamicImageView`](crate::DynamicImageView) or channels can't be converted into an [`ImageView`](crate::ImageView)
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IncompatibleImageError {
    #[error(transparent)]
    Channel(#[from] IncompatibleChannelError),
    #[error("The channels have different sizes: {a:?} and {b:?}")]
    MixedImageSizes {
        a: (NonZeroU32, NonZeroU32),
        b: (NonZeroU32, NonZeroU32),
    },
    #[error(transparent)]
    ChannelCount(#[from] IncompatibleChannelCount),
}

/// The image doesn't have the required number of channels
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("The image has {actual} channels, but exactly {expected} are required")]
#[non_exhaustive]
pub struct IncompatibleChannelCount {
    pub expected: NonZeroU8,
    pub actual: NonZeroUsize,
}

/// A channel can't be viewed as the target pixel type
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "The channel ({actual}, aligned={aligned}) doesn't match {target_pixel_type} ({required:?})"
)]
#[non_exhaustive]
pub struct IncompatibleChannelError {
    pub actual: PixelLayout,
    /// Layout of the target pixel type, or `None` if there is no single target (e.g. a conversion,
    /// which supports several pixel types, but none with the `actual` layout)
    pub required: Option<PixelLayout>,
    /// Whether the buffer is aligned for the primitive of the target pixel type
    pub aligned: bool,
    pub target_pixel_type: &'static str,
}

impl IncompatibleChannelError {
    /// For conversions outside of this crate, which can't view a channel
    #[must_use]
    pub const fn new(
        actual: PixelLayout,
        required: Option<PixelLayout>,
        aligned: bool,
        target_pixel_type: &'static str,
    ) -> Self {
        Self {
            actual,
            required,
            aligned,
            target_pixel_type,
        }
    }
}

/// Memory layout of a pixel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PixelLayout {
    pub elements: NonZeroU8,
    /// Size of a single element (primitive) of a pixel
    pub bytes_per_primitive: NonZeroUsize,
}

impl PixelLayout {
    #[must_use]
    pub const fn new(elements: NonZeroU8, bytes_per_primitive: NonZeroUsize) -> Self {
        Self {
            elements,
            bytes_per_primitive,
        }
    }

    pub(crate) fn of<TP: PixelType>() -> Self {
        Self::of_primitive::<TP::Primitive>(TP::ELEMENTS)
    }

    /// Layout of `pixel_elements` primitives `T` per pixel
    pub(crate) fn of_primitive<T: PixelTypePrimitive>(pixel_elements: NonZeroU8) -> Self {
        Self::new(pixel_elements, crate::channel::bytes_per_primitive::<T>())
    }
}

impl Display for PixelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pixel_elements={}, bytes_per_primitive={}",
            self.elements, self.bytes_per_primitive
        )
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone)]
#[error("The multiplication of the dimensions overflows `usize`")]
pub struct OverflowBufferLenError {
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
    channels: NonZeroU8,
}

impl OverflowBufferLenError {
    pub(crate) const fn new(
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        channels: NonZeroU8,
    ) -> Self {
        Self {
            width,
            height,
            pixel_elements,
            channels,
        }
    }
}
