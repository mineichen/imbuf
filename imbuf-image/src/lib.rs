//! Conversions between [`imbuf`] buffers and the [`image`] crate.
//!
//! This crate is the standalone home of the `image` integration that used to
//! live in `imbuf::external`. It depends on `imbuf` and provides the value types
//! as well as the conversions connecting the two crates.
//!
//! Only the newest release of `imbuf` and the newest supported version of
//! `image` are covered. If you need another combination (for example a different
//! `image` version, or a version matching an older `imbuf` release), implement
//! the same conversions yourself — this crate is the template. See the
//! repository `README.md`.

use std::num::NonZeroU8;

mod from_imbuf;
mod into_imbuf;
mod ref_image;

pub use from_imbuf::{
    ImbufAsRustImage, ImbufAsRustImageMut, ImbufIntoRustImage, IncompatibleLayoutError,
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
