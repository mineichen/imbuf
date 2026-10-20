#![doc = include_str!("../README.md")]

use std::num::NonZeroU8;

mod from_imbuf;
mod into_imbuf;
mod ref_image;

pub use from_imbuf::{
    ImbufAsRustImage, ImbufAsRustImageMut, ImbufIntoRustImage, ImbufTryIntoRustImage,
    IncompatibleLayoutError,
};
pub use into_imbuf::{ImageToImbuf, IntoDynamicImageError, IntoDynamicImageErrorReason};
pub use ref_image::{DynamicRefImage, DynamicRefImageMut};

/// The buffer length did not match the image dimensions.
#[derive(Debug, thiserror::Error)]
#[error("The image has a wrong length. Expected {expected}, got {actual}")]
pub struct IncompatibleBufferSize {
    pub expected: usize,
    pub actual: usize,
}

fn non_zero_u8(value: usize) -> NonZeroU8 {
    assert!(value <= 255, "usize must be less than 256");
    #[allow(clippy::cast_possible_truncation)]
    NonZeroU8::new(value as u8).unwrap()
}
