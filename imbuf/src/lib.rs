#![doc = include_str!("../README.md")]

use std::num::NonZeroU8;

mod arc;
mod channel;
mod dynamic;
mod error;
mod image;
mod lifetime;
mod pixel;
mod shared_vec;
mod vec;
mod view;

pub use channel::{ImageChannel, ImageChannelBase, ImageChannelVTable, ImageChannelView};
pub use dynamic::{
    DynamicImage, DynamicImageBase, DynamicImageChannel, DynamicImageChannelBase,
    DynamicImageChannelView, DynamicImageView,
};
pub use error::{
    IncompatibleChannel, IncompatibleChannelCount, IncompatibleChannelError, IncompatibleImage,
    IncompatibleImageError, InvalidLenError, OverflowBufferLenError, PixelLayout,
};
pub use image::{Image, ImageBase, ImageView};
pub use lifetime::{Borrowed, BufferLifetime, Owned};
pub use pixel::{PixelType, PixelTypePrimitive};

const fn unwrap_usize_to_nonzero_u8(value: usize) -> NonZeroU8 {
    assert!(value <= 255, "usize must be less than 256");
    #[allow(clippy::cast_possible_truncation)]
    NonZeroU8::new(value as u8).unwrap()
}
