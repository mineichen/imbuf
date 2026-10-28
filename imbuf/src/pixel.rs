use std::{num::NonZeroU8, sync::Arc};

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
pub trait PixelType: Clone + Sized + Send + Sync + 'static + seal::SealedPixelType {
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

pub(crate) fn flat_arc<TP: PixelType>(input: Arc<[TP]>) -> Arc<[TP::Primitive]> {
    let elements = usize::from(TP::ELEMENTS.get());
    let len = input.len() * elements;
    let ptr = Arc::into_raw(input).cast::<TP::Primitive>();

    // Safety: TP::Primitive is expected to be an aligned fraction of TP
    unsafe { Arc::from_raw(core::ptr::slice_from_raw_parts(ptr, len)) }
}
pub(crate) fn flat_vec<TP: PixelType>(input: Vec<TP>) -> Vec<TP::Primitive> {
    let (ptr, len, cap) = input.into_raw_parts();
    let ptr = ptr.cast::<TP::Primitive>();
    // Multiplying with `TP::ELEMENTS` cannot overflow, as a `Vec<TP>` longer than
    // `usize::MAX / TP::ELEMENTS` (a multiple of it in bytes) couldn't be allocated
    let elements = usize::from(TP::ELEMENTS.get());
    let len = len * elements;
    let cap = cap * elements;

    // Safety: TP::Primitive is expected to be an aligned fraction of TP
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

mod seal {
    use std::num::NonZeroUsize;

    use crate::PixelTypePrimitive;

    /// `PixelType` is only implemented for primitives and arrays of primitives.
    /// This crate casts between buffers of pixels and buffers of their primitives,
    /// which is only sound for these layouts.
    pub trait SealedPixelType {
        const NZ_PIXEL_ELEMENTS: NonZeroUsize;
    }
    impl<T: PixelTypePrimitive> SealedPixelType for T {
        const NZ_PIXEL_ELEMENTS: NonZeroUsize = NonZeroUsize::MIN;
    }
    impl<T: PixelTypePrimitive, const PIXEL_ELEMENTS: usize> SealedPixelType for [T; PIXEL_ELEMENTS] {
        const NZ_PIXEL_ELEMENTS: NonZeroUsize = {
            // 1 should be invalid too
            if PIXEL_ELEMENTS <= 1 {
                panic!("PIXEL_ELEMENTS should be at least 2")
            }
            NonZeroUsize::new(PIXEL_ELEMENTS).unwrap()
        };
    }
}
