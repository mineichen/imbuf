use std::{
    any::type_name,
    borrow::Borrow,
    cmp::PartialEq,
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    ptr, slice,
    sync::Arc,
};

use bytemuck::{NoUninit, Pod};

use crate::{dynamic::DynamicImageChannel, pixel::PixelType};

/// Represents a plane of a image. In a Planar RGB image, this could be the R, G, or B channel
/// Interleaved RGB images have a single channel, which is a array of pixels (TP: [u8; 3])
// Repr transparent is important for casting between ImageChannel and DynamicImageChannel
#[repr(transparent)]
pub struct ImageChannel<TP: PixelType>(DynamicImageChannel, PhantomData<TP>);

pub trait BorrowableImageChannel:
    Borrow<ImageChannel<Self::Pixel>> + crate::seal::SealedImageChannel
{
    type Pixel: PixelType;
}

impl<T: PixelType> BorrowableImageChannel for ImageChannel<T> {
    type Pixel = T;
}

impl<T: PixelType> BorrowableImageChannel for &ImageChannel<T> {
    type Pixel = T;
}

impl<T: PixelType> BorrowableImageChannel for &mut ImageChannel<T> {
    type Pixel = T;
}

impl<TP: PixelType> Clone for ImageChannel<TP> {
    fn clone(&self) -> Self {
        // Safety: The vtable clones the channel with the same primitive and pixel_elements
        let clone = unsafe { Self::from_dynamic_unchecked(self.0.clone()) };
        clone.debug_assert_layout();
        clone
    }
}

impl<TP: PixelType> PartialEq for ImageChannel<TP> {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions() == other.dimensions() && self.buffer_flat() == other.buffer_flat()
    }
}

impl<TP: PixelType> ImageChannel<TP> {
    /// # Panics
    /// Panics if the buffer size is not compatible with the width and height.
    #[must_use]
    pub fn new_vec(input: Vec<TP>, width: NonZeroU32, height: NonZeroU32) -> Self {
        assert_eq!(
            input.len(),
            calc_pixel_len_packed(width, height),
            "Incompatible Buffer-Size"
        );

        // Cast Vec<TP> to Vec<TP::Primitive>
        let mut input = mem::ManuallyDrop::new(input);
        let ptr = input.as_mut_ptr().cast::<TP::Primitive>();
        let len = input.len() * TP::ELEMENTS.get() as usize;
        let cap = input.capacity() * TP::ELEMENTS.get() as usize;

        // Safety: TP::Primitive is expected to be an aligned fraction of TP
        let cast_input = unsafe { Vec::from_raw_parts(ptr, len, cap) };

        let channel = DynamicImageChannel::from_vec(cast_input, width, height, TP::ELEMENTS);
        // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
        unsafe { Self::from_dynamic_unchecked(channel) }
    }

    /// # Panics
    /// Panics if the buffer size is not compatible with the width and height.
    #[must_use]
    pub fn new_arc(input: Arc<[TP]>, width: NonZeroU32, height: NonZeroU32) -> Self {
        let len = input.len() * TP::ELEMENTS.get() as usize;
        let ptr = Arc::into_raw(input).cast::<TP::Primitive>();

        // Safety: TP::Primitive is expected to be an aligned fraction of TP
        let cast_input = unsafe { Arc::from_raw(ptr::slice_from_raw_parts(ptr, len)) };

        let channel = DynamicImageChannel::from_arc(cast_input, width, height, TP::ELEMENTS);
        // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
        unsafe { Self::from_dynamic_unchecked(channel) }
    }

    /// Don't use this method unless you need a custom storage.
    ///
    /// Use/provide methods like `new_vec()` and `new_arc()` for safe construction.
    /// Untyped storages can be created with [`DynamicImageChannel::from_unsafe`] instead.
    ///
    /// # Safety
    /// The fields must fulfill the invariants of [`DynamicImageChannel::from_unsafe`], except that
    /// the buffer only has to be valid for `TP::Primitive` instead of any `Pod` type.
    /// Furthermore, `ptr` must be aligned for `TP::Primitive` and `vtable.primitive_size()` must
    /// return `size_of::<TP::Primitive>()`.
    pub unsafe fn new_with_vtable(
        ptr: *const TP::Primitive,
        width: NonZeroU32,
        height: NonZeroU32,
        vtable: &'static ImageChannelVTable,
        data: *mut (),
    ) -> Self {
        let unsafe_ch = UnsafeImageChannel::new_with_vtable(
            ptr.cast(),
            width,
            height,
            TP::ELEMENTS,
            vtable,
            data,
        );
        let channel = unsafe { DynamicImageChannel::from_unsafe(unsafe_ch) };
        // Safety: Forwarded to the caller
        unsafe { Self::from_dynamic_unchecked(channel) }
    }

    /// # Safety
    /// `channel` must contain a valid buffer of `TP::Primitive` with `TP::ELEMENTS` per pixel
    pub(crate) const unsafe fn from_dynamic_unchecked(channel: DynamicImageChannel) -> Self {
        Self(channel, PhantomData)
    }

    #[must_use]
    pub fn buffer(&self) -> &[TP] {
        let flat = self.buffer_flat();
        // Safety: The flat buffer contains `len()` pixels with `TP::ELEMENTS` primitives each
        unsafe { slice::from_raw_parts(flat.as_ptr().cast::<TP>(), self.len()) }
    }

    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    pub fn make_mut(&mut self) -> &mut [TP] {
        let len = self.len();
        let flat = self.primitive_make_mut();
        // Safety: The flat buffer contains `len` pixels with `TP::ELEMENTS` primitives each
        unsafe { slice::from_raw_parts_mut(flat.as_mut_ptr().cast::<TP>(), len) }
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<TP> {
        let mut vec = self.into_vec_flat();
        let elements = TP::ELEMENTS.get() as usize;
        if !vec.capacity().is_multiple_of(elements) {
            // Vec<TP> cannot represent the allocation. Boxed slices have an exact capacity.
            vec = vec.into_boxed_slice().into_vec();
        }

        // Cast Vec<TP::Primitive> back to Vec<TP>
        let mut vec = mem::ManuallyDrop::new(vec);
        let ptr = vec.as_mut_ptr().cast::<TP>();
        let len = vec.len() / elements;
        let cap = vec.capacity() / elements;

        // Safety: TP::Primitive is an aligned fraction of TP and capacity is a multiple of TP::ELEMENTS
        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub const fn len_flat(&self) -> usize {
        self.0.len_flat()
    }

    #[must_use]
    pub fn buffer_flat(&self) -> &[TP::Primitive] {
        // Safety: The primitive was checked during construction
        unsafe { self.0.buffer_flat_unchecked() }
    }

    #[must_use]
    pub fn buffer_flat_bytes(&self) -> &[u8]
    where
        TP::Primitive: NoUninit,
    {
        bytemuck::cast_slice(self.buffer_flat())
    }

    pub fn primitive_make_mut(&mut self) -> &mut [TP::Primitive] {
        self.debug_assert_layout();
        unsafe { self.0.primitive_make_mut_unchecked() }
    }

    #[must_use]
    pub fn into_vec_flat(self) -> Vec<TP::Primitive> {
        let raw = DynamicImageChannel::get_unsafe(&self.0);
        let vec_drop: unsafe extern "C-unwind" fn(&mut DynamicImageChannel) =
            crate::vec::drop_vec_channel::<TP::Primitive>;
        // Check if this is a Vec-backed channel
        if ptr::fn_addr_eq(raw.vtable.drop, vec_drop) {
            let ptr = raw.ptr.cast::<TP::Primitive>().cast_mut();
            let capacity = raw.data as usize;
            let len = self.len_flat();
            mem::forget(self);
            // Safety: The Vec-vtable stores the capacity of a Vec<TP::Primitive> in data
            unsafe { Vec::from_raw_parts(ptr, len, capacity) }
        } else {
            self.buffer_flat().to_vec()
        }
    }

    #[must_use]
    pub const fn width(&self) -> NonZeroU32 {
        self.0.width()
    }

    #[must_use]
    pub const fn height(&self) -> NonZeroU32 {
        self.0.height()
    }

    #[must_use]
    pub const fn dimensions(&self) -> (NonZeroU32, NonZeroU32) {
        self.0.dimensions()
    }

    #[must_use]
    pub const fn dimensions_zeroable(&self) -> (u32, u32) {
        self.0.dimensions_zeroable()
    }

    #[must_use]
    pub const fn pixel_elements(&self) -> NonZeroU8 {
        self.0.pixel_elements()
    }

    /// Vtables are expected to keep the layout of their primitive. A violation is a bug in the vtable.
    fn debug_assert_layout(&self) {
        debug_assert!(
            self.0.fits_pixel::<TP>(),
            "The vtable changed the layout or alignment of the primitive"
        );
    }
}

impl<TP: PixelType> TryFrom<DynamicImageChannel> for ImageChannel<TP>
where
    TP::Primitive: Pod,
{
    type Error = DynamicImageChannel;

    fn try_from(value: DynamicImageChannel) -> Result<Self, Self::Error> {
        if value.fits_pixel::<TP>() {
            // Safety: Size, alignment and pixel_elements are checked. Pod accepts any bit pattern
            Ok(unsafe { Self::from_dynamic_unchecked(value) })
        } else {
            Err(value)
        }
    }
}

impl<'a, TP: PixelType> TryFrom<&'a DynamicImageChannel> for &'a ImageChannel<TP>
where
    TP::Primitive: Pod,
{
    type Error = ();

    fn try_from(value: &'a DynamicImageChannel) -> Result<Self, Self::Error> {
        value.try_cast().ok_or(())
    }
}

impl<'a, TP: PixelType> TryFrom<&'a mut DynamicImageChannel> for &'a mut ImageChannel<TP>
where
    TP::Primitive: Pod,
{
    type Error = ();

    fn try_from(value: &'a mut DynamicImageChannel) -> Result<Self, Self::Error> {
        value.try_cast_mut().ok_or(())
    }
}

/// Requires `Pod`, because a `DynamicImageChannel` can be viewed as any `Pod` primitive of the same size
impl<TP: PixelType> From<ImageChannel<TP>> for DynamicImageChannel
where
    TP::Primitive: Pod,
{
    fn from(value: ImageChannel<TP>) -> Self {
        value.0
    }
}

impl<TP: PixelType> Debug for ImageChannel<TP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageChannel")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("pixel", &type_name::<TP::Primitive>())
            .field("pixel_elements", &self.pixel_elements().get())
            .finish()
    }
}

/// `VTable` for [`DynamicImageChannel`] and [`ImageChannel`]
///
/// The vtable itself is untyped, but its functions are usually generic over the primitive of the
/// buffer. All functions use the "C-unwind" ABI, so panics propagate instead of aborting.
///
/// The functions receive a [`DynamicImageChannel`], whose raw parts are accessible via
/// [`DynamicImageChannel::get_unsafe`] and [`DynamicImageChannel::get_unsafe_mut`]. For channels
/// of non-`Pod` primitives (only reachable through [`ImageChannel`]), the buffer is only valid for
/// the primitive the vtable was built for.
#[repr(C)]
pub struct ImageChannelVTable {
    /// Returns a channel with the same dimensions, `pixel_elements` and content, which can be used
    /// independently. The buffer can be shared until `make_mut` is called.
    pub clone: unsafe extern "C-unwind" fn(&DynamicImageChannel) -> DynamicImageChannel,
    /// Ensures that the buffer is exclusively owned by the channel, so it can be mutated.
    /// The entire channel can be replaced (assigning a `DynamicImageChannel` drops the old one).
    /// The new buffer has to be aligned for the primitive.
    pub make_mut: unsafe extern "C-unwind" fn(&mut DynamicImageChannel),
    /// Releases the buffer and `data`. The channel is not used afterwards.
    pub drop: unsafe extern "C-unwind" fn(&mut DynamicImageChannel),
    /// Size of the primitive in bytes. It has to return the same value on every call.
    pub primitive_size: extern "C-unwind" fn() -> NonZeroUsize,
}

/// Raw parts of a channel. This is only relevant when implementing a custom storage.
///
/// In contrast to [`DynamicImageChannel`], `UnsafeImageChannel` has no guarantee about the
/// consistency of its fields. It doesn't implement `Drop`, so dropping it leaks the buffer.
/// Use [`DynamicImageChannel::from_unsafe`] to take ownership of it.
#[repr(C)]
#[must_use = "Dropping an UnsafeImageChannel leaks its buffer"]

pub struct UnsafeImageChannel {
    pub ptr: *const u8,
    pub width: NonZeroU32,
    pub height: NonZeroU32,
    pub vtable: &'static ImageChannelVTable,
    /// Storage specific data, which has to be cleaned up by `vtable.drop`
    pub data: *mut (),
    pub pixel_elements: NonZeroU8,
}

impl UnsafeImageChannel {
    /// Don't use this method unless you need a custom storage.
    ///
    /// Use/provide methods like `new_vec()` and `new_arc()` for safe construction
    pub const fn new_with_vtable(
        ptr: *const u8,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        vtable: &'static ImageChannelVTable,
        data: *mut (),
    ) -> Self {
        UnsafeImageChannel {
            ptr,
            width,
            height,
            vtable,
            data,
            pixel_elements,
        }
    }

    pub(crate) const fn calc_len_packed(&self) -> usize {
        calc_pixel_len_packed(self.width, self.height)
    }

    pub(crate) const fn calc_len_flat(&self) -> usize {
        calc_pixel_len_flat(self.width, self.height, self.pixel_elements)
    }
}

const fn calc_pixel_len_packed(width: NonZeroU32, height: NonZeroU32) -> usize {
    #[allow(clippy::cast_possible_truncation)]
    let width_usize = width.get() as usize;
    #[allow(clippy::cast_possible_truncation)]
    let height_usize = height.get() as usize;

    width_usize * height_usize
}

pub(crate) const fn calc_pixel_len_flat(
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
) -> usize {
    calc_pixel_len_packed(width, height) * pixel_elements.get() as usize
}

/// `ImageChannelVTable::primitive_size` for the primitive `T`
pub(crate) extern "C-unwind" fn primitive_size<T>() -> NonZeroUsize {
    const { NonZeroUsize::new(size_of::<T>()).expect("Zero sized primitives are not supported") }
}

// Workaround inability to have static which uses Outer Generics
pub(crate) trait ChannelFactory<T> {
    const VTABLE: &'static ImageChannelVTable;
}

#[cfg(test)]
mod tests {
    use crate::{DynamicImage, Image};

    use super::*;

    #[test]
    fn try_from_dynamic_channel() {
        let size = NonZeroU32::MIN;
        let data = vec![[1u8, 2, 3]];
        let ch = ImageChannel::<[u8; 3]>::new_vec(data, size, size);
        let image: Image<[u8; 3], 1> = [ch].try_into().unwrap();
        let dyn_image: DynamicImage = image.into();
        let (restored_channel, mut rest) =
            dyn_image.into_mapped_channels(|x| ImageChannel::<[u8; 3]>::try_from(x).unwrap());
        assert_eq!(None, rest.next());
        assert_eq!(restored_channel.buffer(), &[[1u8, 2, 3]]);
        Image::<[u8; 3], 1>::try_from([restored_channel]).unwrap();
    }

    #[test]
    fn try_from_dynamic_channel_checks_primitive_and_pixel_elements() {
        let size = NonZeroU32::MIN;
        let dynamic = DynamicImageChannel::from(ImageChannel::<u16>::new_vec(vec![1], size, size));

        let dynamic = ImageChannel::<u8>::try_from(dynamic).unwrap_err();
        let dynamic = ImageChannel::<[u8; 2]>::try_from(dynamic).unwrap_err();
        let dynamic = ImageChannel::<[u16; 2]>::try_from(dynamic).unwrap_err();
        let typed = ImageChannel::<u16>::try_from(dynamic).unwrap();
        assert_eq!(typed.buffer(), &[1]);
    }

    #[test]
    fn try_cast_dynamic_channel() {
        let size = NonZeroU32::MIN;
        let data = vec![[1u8, 2, 3]];
        let data_ptr = data.as_ptr() as usize;
        let ch = ImageChannel::<[u8; 3]>::new_vec(data, size, size);
        let image: Image<[u8; 3], 1> = [ch].try_into().unwrap();
        let dyn_image: DynamicImage = image.into();
        let dyn_ch = &dyn_image[0];
        let as_rgb = dyn_ch.try_cast::<[u8; 3]>().unwrap();
        assert_eq!(as_rgb.buffer(), &[[1u8, 2, 3]]);
        assert_eq!(
            as_rgb.buffer().as_ptr() as usize,
            data_ptr,
            "Should share the buffer"
        );

        assert!(dyn_ch.try_cast::<u8>().is_none());
        assert!(dyn_ch.try_cast::<[u8; 4]>().is_none());
        assert!(dyn_ch.try_cast::<[u16; 3]>().is_none());
    }

    #[test]
    fn into_vec_with_capacity_which_is_no_multiple_of_pixel_elements() {
        let size = NonZeroU32::MIN;
        let mut data = Vec::with_capacity(4);
        data.extend_from_slice(&[1u8, 2, 3]);
        let dynamic = DynamicImageChannel::new_vec(data, size, size, NonZeroU8::new(3).unwrap());
        let typed = ImageChannel::<[u8; 3]>::try_from(dynamic).unwrap();
        assert_eq!(typed.into_vec(), vec![[1, 2, 3]]);
    }

    #[test]
    fn miri_create_and_clear_vec_image_channel() {
        let size = 2.try_into().unwrap();
        let image = ImageChannel::<u8>::new_vec(vec![0u8, 64u8, 128u8, 192u8], size, size);
        assert_eq!(image.buffer(), &[0u8, 64u8, 128u8, 192u8]);
    }

    #[test]
    fn miri_to_vec_reuses_pointer() {
        let raw = vec![0u8, 64u8, 128u8, 192u8];
        let pointer = raw[..].as_ptr();
        let size = 2.try_into().unwrap();
        let image = ImageChannel::<u8>::new_vec(raw, size, size);
        let to_vec = image.into_vec();

        // Miri seems to generate clear_vec_channel::<const u8> for each call
        // It works on native x86. Because it's only an optimization, this is good enough
        // VTable is not possible, as ImageChannel is ABI-Stable and multiple dylibs use their own allocator for Vecs
        if !cfg!(miri) {
            assert_eq!(
                to_vec[..].as_ptr(),
                pointer,
                "Should reuse the buffer if it was created by vec"
            );
        }
    }

    #[test]
    fn miri_make_mut_reuses_arc_pointer() {
        let raw = Arc::<[u8]>::from([0u8, 64u8, 128u8, 192u8].as_slice());
        let pointer = raw[..].as_ptr();
        let size = 2.try_into().unwrap();
        let mut image = ImageChannel::<u8>::new_arc(raw, size, size);
        let ptr_mut = image.make_mut();

        assert_eq!(
            ptr_mut[..].as_ptr(),
            pointer,
            "Should reuse the buffer if it was created by arc"
        );
    }

    #[test]
    fn miri_make_mut_doesnt_reuse_arc_pointer_if_not_unique() {
        let raw = Arc::<[u8]>::from([0u8, 64u8, 128u8, 192u8].as_slice());
        let _raw2 = raw.clone();
        let pointer = raw[..].as_ptr();
        let size = 2.try_into().unwrap();
        let mut image = ImageChannel::<u8>::new_arc(raw, size, size);
        let ptr_mut = image.make_mut();

        assert_ne!(
            ptr_mut[..].as_ptr(),
            pointer,
            "Should not reuse the buffer if arc is not unique"
        );
    }

    #[test]
    fn miri_clone_arc_backed_shares_memory() {
        let raw = Arc::<[u8]>::from([0u8, 64u8, 128u8, 192u8].as_slice());
        let pointer = raw[..].as_ptr();
        let size = 2.try_into().unwrap();
        let image = ImageChannel::<u8>::new_arc(raw, size, size);
        let image2 = image.clone();

        assert_eq!(
            image2.buffer().as_ptr(),
            pointer,
            "Should share the buffer if it was created by arc"
        );
    }

    #[test]
    fn miri_clone_from_vec() {
        let raw = vec![0u8, 64u8, 128u8, 192u8];
        let size = 2.try_into().unwrap();
        let image = ImageChannel::<u8>::new_vec(raw, size, size);
        let image2 = image.clone();
        let to_vec = image.into_vec();
        let to_vec2 = image2.into_vec();

        assert_ne!(
            to_vec[..].as_ptr(),
            to_vec2[..].as_ptr(),
            "Should not share the buffer if it was created by vec"
        );
    }

    #[test]
    fn miri_test_shared_arc_u16_channel() {
        let arc: Arc<[u16]> = vec![1].into();
        test_entire_vtable(ImageChannel::<u16>::new_arc(
            arc,
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
    }

    #[test]
    fn miri_test_exclusive_arc_u16_channel() {
        test_entire_vtable(ImageChannel::<u16>::new_arc(
            vec![1u16].into(),
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
    }

    #[test]
    fn miri_test_vec_u16_channel() {
        test_entire_vtable(ImageChannel::<u16>::new_vec(
            vec![1u16],
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
    }

    #[test]
    fn miri_test_vec_rgb16_channel() {
        test_entire_vtable(ImageChannel::<[u16; 3]>::new_vec(
            vec![[1u16, 2u16, 3u16]],
            NonZeroU32::MIN,
            NonZeroU32::MIN,
        ));
    }

    fn test_entire_vtable<TP: PixelType>(mut image: ImageChannel<TP>)
    where
        TP::Primitive: Default + Eq + Debug,
    {
        image.primitive_make_mut()[0] = TP::Primitive::default();
        let clone = image.clone();
        assert_eq!(image.primitive_make_mut()[0], TP::Primitive::default());
        image.primitive_make_mut()[0] = TP::Primitive::default();

        assert_eq!(image, clone);
    }

    #[test]
    fn miri_test_buffer_flat_bytes() {
        let image = ImageChannel::new_vec(vec![42u16], NonZeroU32::MIN, NonZeroU32::MIN);
        assert_eq!(image.buffer_flat_bytes(), &[42u8, 0u8]);
    }
}
