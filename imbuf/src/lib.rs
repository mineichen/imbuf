#![doc = include_str!("../README.md")]

use std::num::NonZeroU8;

mod arc;
mod channel;
mod dynamic_channel;
mod dynamic_image;
mod error;
mod image;
mod pixel;
mod shared_vec;
mod transpose;
mod vec;
mod view;
mod vtable;

pub use channel::{ImageChannel, ImageChannelView};
pub use dynamic_channel::{DynamicImageChannel, DynamicImageChannelView};
pub use dynamic_image::{DynamicImage, DynamicImageView};
pub use error::{
    IncompatibleChannel, IncompatibleChannelCount, IncompatibleChannelError, IncompatibleImage,
    IncompatibleImageError, InvalidLenError, OverflowBufferLenError, PixelLayout,
};
pub use image::{Image, ImageView};
pub use pixel::{PixelType, PixelTypePrimitive};
pub use vtable::ImageChannelVTable;

const fn unwrap_usize_to_nonzero_u8(value: usize) -> NonZeroU8 {
    assert!(value <= 255, "usize must be less than 256");
    #[allow(clippy::cast_possible_truncation)]
    NonZeroU8::new(value as u8).unwrap()
}
