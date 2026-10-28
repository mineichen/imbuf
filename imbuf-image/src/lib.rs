#![doc = include_str!("../README.md")]

mod from_imbuf;
mod into_imbuf;
mod ref_image;

pub use from_imbuf::{ImbufAsImage, ImbufAsImageMut, ImbufIntoImage};
pub use into_imbuf::{ImageIntoImbuf, IntoImbufError, IntoImbufErrorReason};
pub use ref_image::{DynamicRefImage, DynamicRefImageMut};

/// The buffer length did not match the image dimensions.
#[derive(Debug, thiserror::Error)]
#[error("The image has a wrong length. Expected {expected}, got {actual}")]
pub struct IncompatibleBufferSize {
    pub expected: usize,
    pub actual: usize,
}
