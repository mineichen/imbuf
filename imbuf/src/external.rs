#[cfg(feature = "image_0_25")]
mod image_0_25;

#[cfg(feature = "image_0_25")]
pub use image_0_25::*;

#[deprecated(
    since = "0.3.4",
    note = "Use the `imbuf-image` crate instead, e.g. `imbuf_image::IncompatibleBufferSize`"
)]
#[derive(Debug, thiserror::Error)]
#[error("The image has a wrong length. Expected {expected}, got {actual}")]
#[cfg(feature = "image_0_25")]
pub struct IncompatibleBufferSize {
    pub expected: usize,
    pub actual: usize,
}
