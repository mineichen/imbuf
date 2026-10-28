#![doc = include_str!("../README.md")]

use std::num::NonZeroU8;

mod arc;
mod channel;
mod dynamic;
mod error;
mod image;
mod pixel;
mod shared_vec;
mod vec;
mod view;

pub use channel::{ImageChannel, ImageChannelVTable, ImageChannelView};
pub use dynamic::{DynamicImage, DynamicImageChannel, DynamicImageChannelView, DynamicImageView};
pub use error::{
    IncompatibleChannel, IncompatibleChannelCount, IncompatibleChannelError, IncompatibleImage,
    IncompatibleImageError, InvalidLenError, OverflowBufferLenError, PixelLayout,
};
pub use image::{Image, ImageView};
pub use pixel::{PixelType, PixelTypePrimitive};

const fn unwrap_usize_to_nonzero_u8(value: usize) -> NonZeroU8 {
    assert!(value <= 255, "usize must be less than 256");
    #[allow(clippy::cast_possible_truncation)]
    NonZeroU8::new(value as u8).unwrap()
}
