#[cfg(feature = "image_0_25")]
mod image_0_25;

#[cfg(feature = "image_0_25")]
pub use image_0_25::*;

#[derive(Debug, thiserror::Error)]
#[error("The image has a wrong length. Expected {expected}, got {actual}")]
#[cfg(feature = "image_0_25")]
pub struct IncompatibleBufferSize {
    pub expected: usize,
    pub actual: usize,
}
