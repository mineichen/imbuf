use std::{
    borrow::Borrow,
    fmt::{self, Debug, Formatter},
    num::{NonZeroU8, NonZeroU32},
    sync::Arc,
};

use crate::{
    dynamic::DynamicImageChannel,
    pixel::{DynamicSize, PixelType, PixelTypePrimitive, RuntimePixelType},
    pixel_elements::PixelSize,
};

/// Represents a plane of a image. In a Planar RGB image, this could be the R, G, or B channel
/// Interleaved RGB images have a single channel, which is a array of pixels (TP: [u8; 3])
// Repr transparent is important for casting between ImageChannel and UnsafeImageChannel
#[repr(transparent)]
pub struct ImageChannel<TP: RuntimePixelType>(UnsafeImageChannel<TP::Primitive>);

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

impl<TP: RuntimePixelType> Clone for ImageChannel<TP> {
    fn clone(&self) -> Self {
        Self(unsafe { (self.0.vtable.clone)(&self.0) })
    }
}

impl<TP> PartialEq for ImageChannel<TP>
where
    TP: RuntimePixelType,
    TP::Primitive: std::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.width == other.0.width
            && self.0.height == other.0.height
            && self.0.pixel_elements == other.0.pixel_elements
            && self.buffer_flat() == other.buffer_flat()
    }
}

impl<TP> ImageChannel<TP>
where
    TP: PixelType,
    TP::Primitive: Clone,
{
    /// # Panics
    /// Panics if the buffer size is not compatible with the width and height.
    #[must_use]
    pub fn new_vec(mut input: Vec<TP>, width: NonZeroU32, height: NonZeroU32) -> Self {
        assert_eq!(
            input.len(),
            calc_pixel_len_packed(width, height),
            "Incompatible Buffer-Size"
        );

        // Cast Vec<TP> to Vec<TP::Primitive>
        let len = input.len();
        let cap = input.capacity();

        let ptr = input.as_mut_ptr().cast::<TP::Primitive>();
        let len = len * TP::ELEMENTS.get() as usize;
        let cap = cap * TP::ELEMENTS.get() as usize;
        std::mem::forget(input);

        // Safety: TP::Primitive is expected to be an aligned fraction of TP
        let cast_input = unsafe { Vec::from_raw_parts(ptr, len, cap) };

        Self(UnsafeImageChannel::new_vec(
            cast_input,
            width,
            height,
            TP::PixelSize::default().get(),
        ))
    }

    pub fn new_arc(input: Arc<[TP]>, width: NonZeroU32, height: NonZeroU32) -> Self {
        let len = input.len();
        let ptr = Arc::into_raw(input).cast::<TP::Primitive>();
        let len = len * TP::ELEMENTS.get() as usize;

        // Safety: TP::Primitive is expected to be an aligned fraction of TP
        let cast_input = unsafe { Arc::from_raw(std::ptr::slice_from_raw_parts(ptr, len)) };

        Self(UnsafeImageChannel::new_arc(
            cast_input,
            width,
            height,
            TP::PixelSize::default().get(),
        ))
    }

    #[must_use]
    pub const fn buffer(&self) -> &[TP] {
        unsafe { std::slice::from_raw_parts(self.0.ptr.cast::<TP>(), self.len()) }
    }

    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.0.calc_len_packed()
    }

    pub fn make_mut(&mut self) -> &mut [TP] {
        unsafe {
            (self.0.vtable.make_mut)(&mut self.0);
            let len = self.len_flat();
            let len = len / TP::ELEMENTS.get() as usize;
            std::slice::from_raw_parts_mut(self.0.ptr as *mut TP, len)
        }
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<TP>
    where
        TP: Clone,
    {
        // Get Vec<TP::Primitive> using the base implementation
        let vec_drop: unsafe extern "C" fn(&mut UnsafeImageChannel<TP::Primitive>) =
            crate::vec::clear_vec_channel::<TP::Primitive>;
        let mut vec = if std::ptr::fn_addr_eq(self.0.vtable.drop, vec_drop) {
            let size = self.len_flat();
            let result =
                unsafe { Vec::from_raw_parts(self.0.ptr.cast_mut(), size, self.0.data as usize) };
            std::mem::forget(self);
            result
        } else {
            let len = self.len_flat();
            let buf = unsafe { std::slice::from_raw_parts(self.0.ptr, len) };
            buf.to_vec()
        };

        // Cast Vec<TP::Primitive> back to Vec<TP>
        let ptr = vec.as_mut_ptr().cast::<TP>();
        let len = vec.len() / TP::ELEMENTS.get() as usize;
        let cap = vec.capacity() / TP::ELEMENTS.get() as usize;
        std::mem::forget(vec);

        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    /// Don't use this method unless you need a custom image.
    ///
    /// Use/provide methods like `new_vec()` and `new_arc()` for safe construction
    ///
    /// # Safety
    /// The vtable must be able to cleanup the fields
    pub unsafe fn new_with_vtable(
        ptr: *const TP::Primitive,
        width: NonZeroU32,
        height: NonZeroU32,
        vtable: &'static ImageChannelVTable<TP::Primitive>,
        data: *mut (),
    ) -> Self
    where
        TP::Primitive: Send + Sync,
    {
        unsafe {
            Self(UnsafeImageChannel::new_with_vtable(
                ptr,
                width,
                height,
                TP::PixelSize::default().get(),
                vtable,
                data,
            ))
        }
    }
}

impl<TP: PixelType> TryFrom<DynamicImageChannel> for ImageChannel<TP> {
    type Error = DynamicImageChannel;

    fn try_from(value: DynamicImageChannel) -> Result<Self, Self::Error> {
        let typed = <TP::Primitive as PixelTypePrimitive>::try_from_dynamic_image(value)?;

        if typed.0.pixel_elements == TP::ELEMENTS {
            Ok(ImageChannel(typed.0))
        } else {
            Err(<TP::Primitive as PixelTypePrimitive>::into_runtime_channel(
                typed,
            ))
        }
    }
}

impl<TP: PixelType> TryFrom<&DynamicImageChannel> for &ImageChannel<TP> {
    type Error = ();

    fn try_from(value: &DynamicImageChannel) -> Result<Self, Self::Error> {
        let typed =
            <TP::Primitive as PixelTypePrimitive>::try_from_dynamic_image_ref(value).ok_or(())?;

        if typed.0.pixel_elements == TP::ELEMENTS {
            // Safety: ImageChannel is repr(transparent), so we are allowed to transmute between them
            Ok(unsafe { std::mem::transmute(&typed.0) })
        } else {
            Err(())
        }
    }
}

impl<TP: PixelType> TryFrom<&mut DynamicImageChannel> for &mut ImageChannel<TP> {
    type Error = ();

    fn try_from(value: &mut DynamicImageChannel) -> Result<Self, Self::Error> {
        let typed =
            <TP::Primitive as PixelTypePrimitive>::try_from_dynamic_image_mut(value).ok_or(())?;

        if typed.0.pixel_elements == TP::ELEMENTS {
            // Safety: ImageChannel is repr(transparent), so we are allowed to transmute between them
            Ok(unsafe { std::mem::transmute(&mut typed.0) })
        } else {
            Err(())
        }
    }
}

impl<TP: PixelType> From<ImageChannel<TP>> for DynamicImageChannel {
    fn from(value: ImageChannel<TP>) -> Self {
        ImageChannel::<DynamicSize<TP::Primitive>>::from(value).into()
    }
}

impl<TP: PixelTypePrimitive> From<ImageChannel<DynamicSize<TP>>> for DynamicImageChannel {
    fn from(value: ImageChannel<DynamicSize<TP>>) -> Self {
        TP::into_runtime_channel(value)
    }
}

impl<TP: PixelType> From<ImageChannel<TP>> for ImageChannel<DynamicSize<TP::Primitive>> {
    fn from(value: ImageChannel<TP>) -> Self {
        ImageChannel(value.0)
    }
}

impl<TP: RuntimePixelType> ImageChannel<TP> {
    /// Create an [`ImageChannel`] from an [`UnsafeImageChannel`] (used internally)
    #[must_use]
    pub(crate) fn from_unsafe_internal(unsafe_channel: UnsafeImageChannel<TP::Primitive>) -> Self {
        Self(unsafe_channel)
    }

    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub const fn len_flat(&self) -> usize {
        self.0.calc_len_flat()
    }

    #[must_use]
    pub const fn buffer_flat(&self) -> &[TP::Primitive] {
        let len = self.len_flat();
        unsafe { std::slice::from_raw_parts(self.0.ptr, len) }
    }

    #[must_use]
    pub const fn buffer_flat_bytes(&self) -> &[u8] {
        let len = self.len_flat();
        unsafe {
            std::slice::from_raw_parts(
                self.0.ptr.cast(),
                len * std::mem::size_of::<TP::Primitive>(),
            )
        }
    }

    pub fn primitive_make_mut(&mut self) -> &mut [TP::Primitive] {
        unsafe {
            (self.0.vtable.make_mut)(&mut self.0);
            let len = self.len_flat();
            std::slice::from_raw_parts_mut(self.0.ptr.cast_mut(), len)
        }
    }

    #[must_use]
    pub fn into_vec_flat(self) -> Vec<TP::Primitive>
    where
        TP::Primitive: Clone,
    {
        let vec_drop: unsafe extern "C" fn(&mut UnsafeImageChannel<TP::Primitive>) =
            crate::vec::clear_vec_channel::<TP::Primitive>;
        // Check if this is a Vec-backed channel
        if std::ptr::fn_addr_eq(self.0.vtable.drop, vec_drop) {
            let size = self.len_flat();
            let result =
                unsafe { Vec::from_raw_parts(self.0.ptr.cast_mut(), size, self.0.data as usize) };
            std::mem::forget(self);
            result
        } else {
            self.buffer_flat().to_vec()
        }
    }

    #[must_use]
    pub const fn width(&self) -> NonZeroU32 {
        self.0.width
    }

    #[must_use]
    pub const fn height(&self) -> NonZeroU32 {
        self.0.height
    }

    #[must_use]
    pub const fn dimensions(&self) -> (NonZeroU32, NonZeroU32) {
        (self.0.width, self.0.height)
    }

    pub const fn dimensions_zeroable(&self) -> (u32, u32) {
        (self.0.width.get(), self.0.height.get())
    }

    #[must_use]
    pub const fn pixel_elements(&self) -> NonZeroU8 {
        self.0.pixel_elements
    }
}

impl<T: PixelTypePrimitive> ImageChannel<DynamicSize<T>> {
    /// Try to view this dynamically-sized pixel channel as a statically-sized `TP` pixel channel.
    ///
    /// Returns `Some(&ImageChannel<TP>)` if the runtime `pixel_elements` matches `TP::ELEMENTS`,
    /// otherwise returns `None`.
    #[must_use]
    pub fn try_cast<TP: PixelType<Primitive = T>>(&self) -> Option<&ImageChannel<TP>> {
        if self.pixel_elements() == TP::ELEMENTS {
            // Safety: `ImageChannel<...>` is a thin wrapper around `UnsafeImageChannel<T>`.
            // With `TP::Primitive = T`, the representation is identical; `pixel_elements` check
            // ensures the typed view is consistent.
            Some(unsafe { &*(std::ptr::from_ref(self).cast::<ImageChannel<TP>>()) })
        } else {
            None
        }
    }
}

impl<TP: RuntimePixelType> Debug for ImageChannel<TP>
where
    TP::Primitive: std::any::Any,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageChannel")
            .field("width", &self.0.width)
            .field("height", &self.0.height)
            .field("pixel", &std::any::type_name::<TP::Primitive>())
            .field("pixel_elements", &self.0.pixel_elements.get())
            .finish()
    }
}

unsafe impl<TP: RuntimePixelType> Send for ImageChannel<TP> where TP::Primitive: Send {}
unsafe impl<TP: RuntimePixelType> Sync for ImageChannel<TP> where TP::Primitive: Sync {}

/// `VTable` for [`ImageChannel`] and [`UnsafeImageChannel`]
#[repr(C)]
pub struct ImageChannelVTable<T: 'static> {
    pub clone: unsafe extern "C" fn(&UnsafeImageChannel<T>) -> UnsafeImageChannel<T>,
    pub make_mut: unsafe extern "C" fn(&mut UnsafeImageChannel<T>),
    pub drop: unsafe extern "C" fn(&mut UnsafeImageChannel<T>),
}

/// This is only relevant when implementing a custom storage
/// T is usually a `PixelTypePrimitive`, but it is not enforced by the type system
#[repr(C)]
pub struct UnsafeImageChannel<T: 'static> {
    pub ptr: *const T,
    pub width: NonZeroU32,
    pub height: NonZeroU32,
    pub vtable: &'static ImageChannelVTable<T>,
    // Has to be cleaned up by clear proc too
    pub data: *mut (),
    pub pixel_elements: NonZeroU8,
}

impl<T: 'static> UnsafeImageChannel<T> {
    /// Don't use this method unless you need a custom image.
    ///
    /// Use/provide methods like `new_vec()` and `new_arc()` for safe construction
    ///
    /// # Safety
    /// The vtable must be able to cleanup the fields
    pub unsafe fn new_with_vtable(
        ptr: *const T,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
        vtable: &'static ImageChannelVTable<T>,
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

    pub(crate) const fn buffer_flat(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.calc_len_flat()) }
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

impl<T> Drop for UnsafeImageChannel<T> {
    fn drop(&mut self) {
        if self.ptr as usize != 0 {
            unsafe { (self.vtable.drop)(self) };
        }
    }
}

// Workaround inability to have static which uses Outer Generics
pub(crate) trait ChannelFactory<T: 'static> {
    const VTABLE: &'static ImageChannelVTable<T>;
}

#[cfg(test)]
mod tests {
    use crate::{DynamicImage, Image};

    use super::*;

    #[test]
    fn try_cast_dynamic_size_channel() {
        let size = NonZeroU32::MIN;
        let data = vec![[1u8, 2, 3]];
        let data_ptr = data.as_ptr() as usize;
        let ch = ImageChannel::<[u8; 3]>::new_vec(data, size, size);
        let image: Image<[u8; 3], 1> = [ch].try_into().unwrap();
        let dyn_image: DynamicImage = image.into();
        let DynamicImageChannel::U8(dyn_u8_ch) = &dyn_image[0] else {
            panic!("Expected DynamicImageChannel::U8");
        };
        let as_rgb = dyn_u8_ch.try_cast::<[u8; 3]>().unwrap();
        assert_eq!(as_rgb.buffer(), &[[1u8, 2, 3]]);
        assert_eq!(
            as_rgb.buffer().as_ptr() as usize,
            data_ptr,
            "Should share the buffer"
        );

        assert!(dyn_u8_ch.try_cast::<u8>().is_none());
        assert!(dyn_u8_ch.try_cast::<[u8; 4]>().is_none());
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

    fn test_entire_vtable<TP: RuntimePixelType>(mut image: ImageChannel<TP>)
    where
        TP::Primitive: 'static + Default + Eq + Debug,
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
