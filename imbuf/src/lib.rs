#![doc = include_str!("../README.md")]
#![warn(clippy::pedantic)]

use std::num::NonZeroU8;

mod arc;
mod channel;
mod dynamic;
mod external;
mod image;
mod pixel;
mod pixel_elements;
mod shared_vec;
mod vec;

pub use channel::{BorrowableImageChannel, ImageChannel, ImageChannelVTable, UnsafeImageChannel};
pub use dynamic::{DynamicImage, DynamicImageChannel, IncompatibleImageError};
#[cfg(feature = "image_0_25")]
pub use external::*;
pub use image::{Image, ImageChannels, ImageMut, ImageRef};
pub use pixel::{DynamicSize, PixelType, PixelTypePrimitive};

#[deprecated(
    since = "0.3.0",
    note = "Use Image<T, 1> instead... This crate makes no assumption about the intent of channels"
)]
pub type LumaImage<T> = Image<T, 1>;
#[deprecated(
    since = "0.3.0",
    note = "Use Image<[T; 3], 1> instead... This crate makes no assumption about the intent of channels"
)]
pub type RgbImageInterleaved<T> = Image<[T; 3], 1>;
#[deprecated(
    since = "0.3.0",
    note = "Use Image<[T; 4], 1> instead... This crate makes no assumption about the intent of channels"
)]
pub type RgbaImageInterleaved<T> = Image<[T; 4], 1>;
#[deprecated(
    since = "0.3.0",
    note = "Use Image<T, 3> instead... This crate makes no assumption about the intent of channels"
)]
pub type RgbImagePlanar<T> = Image<T, 3>;
#[deprecated(
    since = "0.3.0",
    note = "Use Image<T, 4> instead... This crate makes no assumption about the intent of channels"
)]
pub type RgbaImagePlanar<T> = Image<T, 4>;

const fn unwrap_usize_to_nonzero_u8(value: usize) -> NonZeroU8 {
    assert!(value <= 255, "usize must be less than 256");
    #[allow(clippy::cast_possible_truncation)]
    NonZeroU8::new(value as u8).unwrap()
}

mod seal {
    use crate::{ImageChannel, PixelType};

    /// Allows to forbid external implementations to add new primitives
    /// This crate heavily relies on casting between primitive pointers
    pub trait SealedPrimitive {}
    impl SealedPrimitive for u8 {}
    impl SealedPrimitive for u16 {}
    impl SealedPrimitive for f32 {}

    pub trait SealedImageChannel {}
    impl<T: PixelType> SealedImageChannel for ImageChannel<T> {}
    impl<T: PixelType> SealedImageChannel for &ImageChannel<T> {}
    impl<T: PixelType> SealedImageChannel for &mut ImageChannel<T> {}
}
