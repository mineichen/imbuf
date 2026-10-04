use std::num::NonZeroU8;

use crate::unwrap_usize_to_nonzero_u8;

pub trait PixelSize: Sized + PartialEq + Clone + Copy + Send + Sync + 'static {
    fn get(&self) -> NonZeroU8;
}

// PIXEL_ELEMENTS is usize, because it's also used to define array lengths. Casting in const is not currently possible
#[derive(Clone, Copy, PartialEq, Default)]
pub struct ComptimeSize<const PIXEL_ELEMENTS: usize>();

impl<const PIXEL_ELEMENTS: usize> PixelSize for ComptimeSize<PIXEL_ELEMENTS> {
    fn get(&self) -> NonZeroU8 {
        const { unwrap_usize_to_nonzero_u8(PIXEL_ELEMENTS) }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct RuntimeSize(pub(crate) NonZeroU8);

impl Default for RuntimeSize {
    fn default() -> Self {
        Self(NonZeroU8::MIN)
    }
}

impl PixelSize for RuntimeSize {
    fn get(&self) -> NonZeroU8 {
        self.0
    }
}
