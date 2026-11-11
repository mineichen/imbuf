use std::{num::NonZeroU8, sync::Arc};

/// Element of a pixel (e.g. `u8` for `[u8; 3]`), which is stored in the buffer of an [`ImageChannel`](crate::ImageChannel).
///
/// This trait can be implemented for custom types. Zero sized types are rejected at compile time.
/// Only primitives which implement [`bytemuck::Pod`] can be converted from and into a
/// [`DynamicImageChannel`](crate::DynamicImageChannel), because the dynamic channel only
/// knows the size of its primitive and allows to reinterpret its bytes.
pub trait PixelTypePrimitive: Copy + PartialEq + Send + Sync + 'static {
    /// Name of the primitive, e.g. `*b"u8\0\0\0\0\0\0"` for `u8`. It's available at runtime
    /// through [`DynamicImageChannelView::type_hint`](crate::DynamicImageChannelView::type_hint),
    /// e.g. to tell `u16` and `i16` channels apart.
    ///
    /// The bytes before the first nul byte are the name, so it has at most 8 bytes and needs no
    /// nul terminator, if it uses all 8 (e.g. `*b"RGBA8888"`). The name has to be valid UTF-8,
    /// which is checked at compile time, when a channel of the primitive is created:
    ///
    /// ```compile_fail,E0080
    /// use std::num::NonZeroU32;
    /// use imbuf::{ImageChannel, PixelTypePrimitive};
    ///
    /// #[derive(Clone, Copy, PartialEq)]
    /// struct Invalid(u8);
    /// impl PixelTypePrimitive for Invalid {
    ///     const TYPE_HINT: [u8; 8] = [0xFF; 8];
    /// }
    ///
    /// let channel = ImageChannel::new_vec(vec![Invalid(0)], NonZeroU32::MIN, NonZeroU32::MIN);
    /// ```
    const TYPE_HINT: [u8; 8];
}

/// Name in `hint` (the bytes before the first nul byte), if it's valid UTF-8. See
/// [`PixelTypePrimitive::TYPE_HINT`].
pub(crate) const fn parse_type_hint(hint: &[u8; 8]) -> Option<&str> {
    let mut len = 0;
    while len < hint.len() && hint[len] != 0 {
        len += 1;
    }
    match std::str::from_utf8(hint.split_at(len).0) {
        Ok(name) => Some(name),
        Err(_) => None,
    }
}

impl PixelTypePrimitive for u8 {
    const TYPE_HINT: [u8; 8] = *b"u8\0\0\0\0\0\0";
}
impl PixelTypePrimitive for i8 {
    const TYPE_HINT: [u8; 8] = *b"i8\0\0\0\0\0\0";
}
impl PixelTypePrimitive for u16 {
    const TYPE_HINT: [u8; 8] = *b"u16\0\0\0\0\0";
}
impl PixelTypePrimitive for i16 {
    const TYPE_HINT: [u8; 8] = *b"i16\0\0\0\0\0";
}
impl PixelTypePrimitive for u32 {
    const TYPE_HINT: [u8; 8] = *b"u32\0\0\0\0\0";
}
impl PixelTypePrimitive for i32 {
    const TYPE_HINT: [u8; 8] = *b"i32\0\0\0\0\0";
}
impl PixelTypePrimitive for u64 {
    const TYPE_HINT: [u8; 8] = *b"u64\0\0\0\0\0";
}
impl PixelTypePrimitive for i64 {
    const TYPE_HINT: [u8; 8] = *b"i64\0\0\0\0\0";
}
impl PixelTypePrimitive for usize {
    const TYPE_HINT: [u8; 8] = *b"usize\0\0\0";
}
impl PixelTypePrimitive for isize {
    const TYPE_HINT: [u8; 8] = *b"isize\0\0\0";
}
impl PixelTypePrimitive for f32 {
    const TYPE_HINT: [u8; 8] = *b"f32\0\0\0\0\0";
}
impl PixelTypePrimitive for f64 {
    const TYPE_HINT: [u8; 8] = *b"f64\0\0\0\0\0";
}

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
pub(crate) const fn flat_slice<TP: PixelType>(input: &[TP]) -> &[TP::Primitive] {
    // Multiplying with `TP::ELEMENTS` cannot overflow, as a slice longer than
    // `usize::MAX / TP::ELEMENTS` (a multiple of it in bytes) cannot exist
    let len = input.len() * (TP::ELEMENTS.get() as usize);

    // Safety: TP::Primitive is expected to be an aligned fraction of TP
    unsafe { core::slice::from_raw_parts(input.as_ptr().cast::<TP::Primitive>(), len) }
}

pub(crate) fn flat_slice_mut<TP: PixelType>(input: &mut [TP]) -> &mut [TP::Primitive] {
    let len = input.len() * usize::from(TP::ELEMENTS.get());

    // Safety: See flat_slice
    unsafe { core::slice::from_raw_parts_mut(input.as_mut_ptr().cast::<TP::Primitive>(), len) }
}

pub(crate) fn flat_vec<TP: PixelType>(input: Vec<TP>) -> Vec<TP::Primitive> {
    let mut input = core::mem::ManuallyDrop::new(input);
    let (ptr, len, cap) = (input.as_mut_ptr(), input.len(), input.capacity());
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
            assert!(PIXEL_ELEMENTS > 1, "PIXEL_ELEMENTS should be at least 2");
            NonZeroUsize::new(PIXEL_ELEMENTS).unwrap()
        };
    }
}
