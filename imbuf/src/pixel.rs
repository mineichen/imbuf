use std::num::NonZeroU8;

/// Element of a pixel (e.g. `u8` for `[u8; 3]`), which is stored in the buffer of an [`ImageChannel`](crate::ImageChannel).
///
/// This trait can be implemented for custom types. Zero sized types are rejected at compile time.
/// Only primitives which implement [`bytemuck::Pod`] can be converted from and into a
/// [`DynamicImageChannel`](crate::DynamicImageChannel), because the dynamic channel only
/// knows the size of its primitive and allows to reinterpret its bytes.
pub trait PixelTypePrimitive: Copy + PartialEq + Send + Sync + 'static {}

impl PixelTypePrimitive for u8 {}
impl PixelTypePrimitive for i8 {}
impl PixelTypePrimitive for u16 {}
impl PixelTypePrimitive for i16 {}
impl PixelTypePrimitive for u32 {}
impl PixelTypePrimitive for i32 {}
impl PixelTypePrimitive for u64 {}
impl PixelTypePrimitive for i64 {}
impl PixelTypePrimitive for usize {}
impl PixelTypePrimitive for isize {}
impl PixelTypePrimitive for f32 {}
impl PixelTypePrimitive for f64 {}

/// A pixel, which is either a [`PixelTypePrimitive`] (e.g. `u8`) or an array of them (e.g. `[u8; 3]`)
pub trait PixelType: Clone + Sized + Send + Sync + 'static + crate::seal::SealedPixelType {
    type Primitive: PixelTypePrimitive;
    const ELEMENTS: NonZeroU8;
}

impl<T: PixelTypePrimitive> PixelType for T {
    type Primitive = T;
    const ELEMENTS: NonZeroU8 = NonZeroU8::MIN;
}

impl<T: PixelTypePrimitive, const PIXEL_ELEMENTS: usize> PixelType for [T; PIXEL_ELEMENTS] {
    type Primitive = T;
    const ELEMENTS: NonZeroU8 = const { crate::unwrap_usize_to_nonzero_u8(PIXEL_ELEMENTS) };
}
