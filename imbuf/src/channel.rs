use std::{
    any::type_name,
    cmp::PartialEq,
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem,
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    ptr, slice,
};

use bytemuck::{NoUninit, Pod};

use crate::{
    PixelTypePrimitive,
    dynamic::DynamicImageChannelView,
    error::{IncompatibleChannel, IncompatibleChannelError, InvalidLenError},
    pixel::PixelType,
};

/// Represents a plane of a image. In a Planar RGB image, this could be the R, G, or B channel
/// Interleaved RGB images have a single channel, which is a array of pixels (TP: [u8; 3])
///
/// The buffer is valid for the lifetime `'a`. Channels which own their buffer (e.g. created by
/// [`ImageChannel::new_vec`]) are `'static` (see [`ImageChannel`]). Borrowed channels are created
/// with [`ImageChannelView::new_slice`], [`ImageChannelView::view`] and their `mut`
/// counterparts. A borrowed channel can't outlive the buffer:
///
/// ```compile_fail,E0597
/// use std::num::NonZeroU32;
/// use imbuf::{ImageChannelView, ImageChannel};
///
/// let owned: ImageChannel<u8> = {
///     let data = vec![1u8];
///     // `ImageChannelView<'_, u8>` can't be converted into `ImageChannelView<'static, u8>`
///     ImageChannelView::new_slice(&data, NonZeroU32::MIN, NonZeroU32::MIN).unwrap()
/// };
/// ```
// Repr transparent is important for casting between ImageChannelView and DynamicImageChannelView
#[repr(transparent)]
pub struct ImageChannelView<'a, TP: PixelType>(DynamicImageChannelView<'a>, PhantomData<TP>);

/// [`ImageChannelView`] which owns its buffer (or shares it with other owners)
pub type ImageChannel<TP> = ImageChannelView<'static, TP>;

impl<TP: PixelType> Clone for ImageChannelView<'_, TP> {
    fn clone(&self) -> Self {
        // Safety: The vtable clones the channel with the same primitive and pixel_elements
        let clone = unsafe { Self::from_dynamic_unchecked(self.0.clone()) };
        debug_assert_eq!(Ok(()), self.0.ensure_fits_pixel::<TP>());
        debug_assert_eq!(Ok(()), clone.0.ensure_fits_pixel::<TP>());

        clone
    }
}

impl<TP: PixelType> PartialEq for ImageChannelView<'_, TP> {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions() == other.dimensions() && self.buffer_flat() == other.buffer_flat()
    }
}

impl<'a, TP: PixelType> ImageChannelView<'a, TP> {
    /// Borrows `input` without copying. Mutable access (e.g. [`ImageChannelView::make_mut`])
    /// copies the buffer, `input` stays unchanged.
    pub fn new_slice(
        input: &'a [TP],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        DynamicImageChannelView::ensure_valid_len(width, height, input.len())?;
        // Safety: The size is checked
        Ok(unsafe { Self::new_slice_unchecked(input, width, height) })
    }

    /// # Safety
    /// `width * height` mustn't overflow `usize` and has to match `input.len()`
    unsafe fn new_slice_unchecked(input: &'a [TP], width: NonZeroU32, height: NonZeroU32) -> Self {
        debug_assert_eq!(
            DynamicImageChannelView::ensure_valid_len(width, height, input.len()),
            Ok(()),
            "Incompatible Buffer-Size"
        );
        // Safety: The size is valid (precondition) and the buffer is borrowed for 'a
        let channel = unsafe {
            crate::view::slice_channel::<TP::Primitive>(
                crate::pixel::flat_slice(input).as_ptr(),
                width,
                height,
                TP::ELEMENTS,
            )
        };
        // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
        unsafe { Self::from_dynamic_unchecked(channel) }
    }

    /// Mutably borrows `input` without copying. Mutable access (e.g. [`ImageChannelView::make_mut`])
    /// writes into `input`. Clones copy the buffer.
    pub fn new_slice_mut(
        input: &'a mut [TP],
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        DynamicImageChannelView::ensure_valid_len(width, height, input.len())?;
        // Safety: The size is checked and the buffer is exclusively borrowed for 'a
        let channel = unsafe {
            crate::view::slice_mut_channel::<TP::Primitive>(
                crate::pixel::flat_slice_mut(input).as_mut_ptr(),
                width,
                height,
                TP::ELEMENTS,
                ptr::null_mut(),
            )
        };
        // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
        Ok(unsafe { Self::from_dynamic_unchecked(channel) })
    }

    /// Borrows the channel without copying the buffer, like [`ImageChannelView::new_slice`].
    #[must_use]
    pub fn view(&self) -> ImageChannelView<'_, TP> {
        // Safety: The buffer of a channel always matches its dimensions
        unsafe { ImageChannelView::new_slice_unchecked(self.buffer(), self.width(), self.height()) }
    }

    /// Mutably borrows the channel without copying the buffer.
    ///
    /// Mutable access to the view calls [`ImageChannelView::make_mut`] on this channel and
    /// writes into its buffer. Clones of the view copy the buffer.
    #[must_use]
    pub fn view_mut(&mut self) -> ImageChannelView<'_, TP> {
        let (width, height) = self.dimensions();
        let ptr = DynamicImageChannelView::raw_ptr(&self.0);
        let original = ptr::from_mut(&mut self.0);
        // Safety: The channel is exclusively borrowed. The vtable calls its make_mut before writing
        let channel = unsafe {
            crate::view::slice_mut_channel::<TP::Primitive>(
                ptr.cast_mut().cast(),
                width,
                height,
                TP::ELEMENTS,
                original,
            )
        };
        // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
        unsafe { ImageChannelView::from_dynamic_unchecked(channel) }
    }

    /// Don't use this method unless you need a custom storage.
    ///
    /// Use/provide methods like `new_vec()` and `new_arc()` for safe construction.
    /// Untyped storages can be created with [`DynamicImageChannelView::from_raw_parts`] instead.
    ///
    /// # Safety
    /// The fields must fulfill the invariants of [`DynamicImageChannelView::from_raw_parts`]
    /// (e.g. the buffer is valid for `'a`), except that the buffer only has to be valid for
    /// `TP::Primitive` instead of any `Pod` type. Furthermore, `ptr` must be aligned for
    /// `TP::Primitive` and `vtable.bytes_per_primitive()` must return `size_of::<TP::Primitive>()`.
    pub unsafe fn new_with_vtable(
        ptr: *const TP::Primitive,
        width: NonZeroU32,
        height: NonZeroU32,
        vtable: &'static ImageChannelVTable,
        data: *mut (),
    ) -> Self {
        let channel = unsafe {
            DynamicImageChannelView::from_raw_parts(
                ptr.cast(),
                width,
                height,
                TP::ELEMENTS,
                vtable,
                data,
            )
        };
        // Safety: Forwarded to the caller
        unsafe { Self::from_dynamic_unchecked(channel) }
    }

    /// # Safety
    /// `channel` must contain a valid buffer of `TP::Primitive` with `TP::ELEMENTS` per pixel
    pub(crate) const unsafe fn from_dynamic_unchecked(
        channel: DynamicImageChannelView<'a>,
    ) -> Self {
        Self(channel, PhantomData)
    }

    /// Reference version of [`ImageChannelView::from_dynamic_unchecked`]
    ///
    /// # Safety
    /// See [`ImageChannelView::from_dynamic_unchecked`]
    pub(crate) unsafe fn from_dynamic_ref_unchecked<'b>(
        channel: &'b DynamicImageChannelView<'a>,
    ) -> &'b Self {
        debug_assert_eq!(Ok(()), channel.ensure_fits_pixel::<TP>());
        // Safety: `ImageChannelView` is repr(transparent) over `DynamicImageChannelView`.
        // The layout is forwarded to the caller
        unsafe { &*ptr::from_ref(channel).cast::<Self>() }
    }

    /// Mutable reference version of [`ImageChannelView::from_dynamic_unchecked`]
    ///
    /// # Safety
    /// See [`ImageChannelView::from_dynamic_unchecked`]
    pub(crate) unsafe fn from_dynamic_mut_unchecked<'b>(
        channel: &'b mut DynamicImageChannelView<'a>,
    ) -> &'b mut Self {
        debug_assert_eq!(Ok(()), channel.ensure_fits_pixel::<TP>());
        // Safety: See `from_dynamic_ref_unchecked`
        unsafe { &mut *ptr::from_mut(channel).cast::<Self>() }
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
        let flat = self.make_mut_flat();
        // Safety: The flat buffer contains `len` pixels with `TP::ELEMENTS` primitives each
        unsafe { slice::from_raw_parts_mut(flat.as_mut_ptr().cast::<TP>(), len) }
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<TP> {
        let mut vec = self.into_vec_flat();
        let elements = TP::ELEMENTS.get() as usize;
        if vec.capacity() % elements != 0 {
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

    pub fn make_mut_flat(&mut self) -> &mut [TP::Primitive] {
        debug_assert_eq!(Ok(()), self.0.ensure_fits_pixel::<TP>());
        unsafe { self.0.make_mut_flat_unchecked() }
    }

    #[must_use]
    pub fn into_vec_flat(self) -> Vec<TP::Primitive> {
        let vtable = DynamicImageChannelView::raw_vtable(&self.0);
        let vec_drop: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>) =
            crate::vec::drop_vec_channel::<TP::Primitive>;
        // Check if this is a Vec-backed channel
        if ptr::fn_addr_eq(vtable.drop, vec_drop) {
            let ptr = DynamicImageChannelView::raw_ptr(&self.0)
                .cast::<TP::Primitive>()
                .cast_mut();
            let capacity = DynamicImageChannelView::raw_data(&self.0) as usize;
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
}

/// Hands back the input in [`IncompatibleChannel`], if it can't be converted. Use `TryFrom`, if the
/// input isn't needed on failure.
impl<'a, TP: PixelType> From<DynamicImageChannelView<'a>>
    for Result<ImageChannelView<'a, TP>, IncompatibleChannel<DynamicImageChannelView<'a>>>
where
    TP::Primitive: Pod,
{
    fn from(value: DynamicImageChannelView<'a>) -> Self {
        match value.ensure_fits_pixel::<TP>() {
            // Safety: Size, alignment and pixel_elements are checked. Pod accepts any bit pattern
            Ok(()) => Ok(unsafe { ImageChannelView::from_dynamic_unchecked(value) }),
            Err(error) => Err(IncompatibleChannel {
                channel: value,
                error,
            }),
        }
    }
}

/// Drops the input on failure. To get it back, convert into
/// `Result<ImageChannelView, IncompatibleChannel<DynamicImageChannelView>>` with `From`/`Into`
/// instead:
///
/// ```
/// use std::num::NonZeroU32;
/// use imbuf::{DynamicImageChannel, ImageChannel, IncompatibleChannel};
///
/// let dynamic = DynamicImageChannel::new_vec(vec![42u8], NonZeroU32::MIN, NonZeroU32::MIN).unwrap();
/// let result: Result<ImageChannel<u16>, IncompatibleChannel<DynamicImageChannel>> = dynamic.into();
/// let IncompatibleChannel { channel, error, .. } = result.unwrap_err();
/// assert_eq!(ImageChannel::<u8>::try_from(channel).unwrap().buffer(), &[42]);
/// println!("{error}");
/// ```
impl<'a, TP: PixelType> TryFrom<DynamicImageChannelView<'a>> for ImageChannelView<'a, TP>
where
    TP::Primitive: Pod,
{
    type Error = IncompatibleChannelError;

    fn try_from(value: DynamicImageChannelView<'a>) -> Result<Self, Self::Error> {
        Result::<Self, IncompatibleChannel<_>>::from(value).map_err(|e| e.error)
    }
}

impl<'a, 'b, TP: PixelType> TryFrom<&'b DynamicImageChannelView<'a>>
    for &'b ImageChannelView<'a, TP>
where
    TP::Primitive: Pod,
{
    type Error = IncompatibleChannelError;

    fn try_from(value: &'b DynamicImageChannelView<'a>) -> Result<Self, Self::Error> {
        value.ensure_fits_pixel::<TP>()?;
        // Safety: The layout is checked and Pod accepts any bit pattern
        Ok(unsafe { ImageChannelView::from_dynamic_ref_unchecked(value) })
    }
}

impl<'a, 'b, TP: PixelType> TryFrom<&'b mut DynamicImageChannelView<'a>>
    for &'b mut ImageChannelView<'a, TP>
where
    TP::Primitive: Pod,
{
    type Error = IncompatibleChannelError;

    fn try_from(value: &'b mut DynamicImageChannelView<'a>) -> Result<Self, Self::Error> {
        value.ensure_fits_pixel::<TP>()?;
        // Safety: The layout is checked and Pod accepts any bit pattern
        Ok(unsafe { ImageChannelView::from_dynamic_mut_unchecked(value) })
    }
}

/// Requires `Pod`, because a `DynamicImageChannelView` can be viewed as any `Pod` primitive of the same size
impl<'a, TP: PixelType> From<ImageChannelView<'a, TP>> for DynamicImageChannelView<'a>
where
    TP::Primitive: Pod,
{
    fn from(value: ImageChannelView<'a, TP>) -> Self {
        value.0
    }
}

impl<TP: PixelType> Debug for ImageChannelView<'_, TP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageChannel")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("pixel", &type_name::<TP::Primitive>())
            .field("pixel_elements", &self.pixel_elements().get())
            .finish()
    }
}

/// `VTable` for [`DynamicImageChannelView`] and [`ImageChannelView`]
///
/// The vtable itself is untyped, but its functions are usually generic over the primitive of the
/// buffer. All functions use the "C-unwind" ABI, so panics propagate instead of aborting.
///
/// The functions receive a [`DynamicImageChannelView`], whose raw parts are accessible through
/// the safe associated functions [`DynamicImageChannelView::raw_ptr`],
/// [`DynamicImageChannelView::raw_data`] and [`DynamicImageChannelView::raw_vtable`]. To mutate
/// a channel, replace it entirely by assigning a new `DynamicImageChannelView` (the old channel
/// is dropped via its vtable).
/// For channels of non-`Pod` primitives (only reachable through [`ImageChannelView`]), the
/// buffer is only valid for the primitive the vtable was built for.
///
/// The functions have to work with channels of any lifetime `'a`, as owned channels can be
/// coerced to shorter lifetimes. `'a` is alive during all calls.
#[repr(C)]
pub struct ImageChannelVTable {
    /// Returns a channel with the same dimensions, `pixel_elements` and content, which can be used
    /// independently. The buffer can be shared until `make_mut` is called.
    pub clone: for<'a> unsafe extern "C-unwind" fn(
        &DynamicImageChannelView<'a>,
    ) -> DynamicImageChannelView<'a>,
    /// Ensures that the buffer is exclusively owned by the channel, so it can be mutated.
    /// The entire channel can be replaced (assigning a `DynamicImageChannelView` drops the old one).
    /// The new buffer has to be aligned for the primitive.
    pub make_mut: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>),
    /// Releases the buffer and `data`. The channel is not used afterwards.
    pub drop: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>),
    /// Size of the primitive in bytes. It has to return the same value on every call.
    pub bytes_per_primitive: extern "C-unwind" fn() -> NonZeroUsize,
}

/// `ImageChannelVTable::bytes_per_primitive` for the primitive `T`
pub(crate) extern "C-unwind" fn bytes_per_primitive<T: PixelTypePrimitive>() -> NonZeroUsize {
    const { NonZeroUsize::new(size_of::<T>()).expect("Zero sized primitives are not supported") }
}

// Workaround inability to have static which uses outer Generics
pub(crate) trait ChannelFactory<T> {
    const VTABLE: &'static ImageChannelVTable;
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use std::sync::Arc;
    use testresult::TestResult;

    use crate::{DynamicImage, DynamicImageChannel, Image};

    use super::*;

    const NZ_1: NonZeroU32 = NonZeroU32::MIN;
    const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

    #[test]
    fn try_from_dynamic_channel() -> TestResult {
        let data = vec![[1u8, 2, 3]];
        let ch = ImageChannel::<[u8; 3]>::new_vec(data, NZ_1, NZ_1)?;
        let image: Image<[u8; 3], 1> = [ch].try_into()?;
        let dyn_image: DynamicImage = image.into();
        let (restored_channel, mut rest) =
            dyn_image.into_mapped_channels(ImageChannel::<[u8; 3]>::try_from);
        assert_eq!(None, rest.next());
        let restored_channel = restored_channel?;
        assert_eq!(restored_channel.buffer(), &[[1u8, 2, 3]]);
        Image::<[u8; 3], 1>::try_from([restored_channel])?;
        Ok(())
    }

    #[test]
    fn try_from_dynamic_channel_checks_primitive_and_pixel_elements() -> TestResult {
        let dynamic = DynamicImageChannel::from(ImageChannel::<u16>::new_vec(vec![1], NZ_1, NZ_1)?);

        assert_matches!(
            ImageChannel::<u8>::try_from(dynamic.clone()),
            Err(IncompatibleChannelError { .. })
        );
        let dynamic = Result::<ImageChannel<u8>, IncompatibleChannel<_>>::from(dynamic)
            .unwrap_err()
            .channel;
        let dynamic = Result::<ImageChannel<[u8; 2]>, IncompatibleChannel<_>>::from(dynamic)
            .unwrap_err()
            .channel;
        let dynamic = Result::<ImageChannel<[u16; 2]>, IncompatibleChannel<_>>::from(dynamic)
            .unwrap_err()
            .channel;
        let typed = ImageChannel::<u16>::try_from(dynamic)?;
        assert_eq!(typed.buffer(), &[1]);
        Ok(())
    }

    #[test]
    fn try_cast_dynamic_channel() -> TestResult {
        let data = vec![[1u8, 2, 3]];
        let data_ptr = data.as_ptr() as usize;
        let ch = ImageChannel::<[u8; 3]>::new_vec(data, NZ_1, NZ_1)?;
        let image: Image<[u8; 3], 1> = [ch].try_into()?;
        let dyn_image: DynamicImage = image.into();
        let dyn_ch = &dyn_image[0];
        let as_rgb = dyn_ch.try_cast::<[u8; 3]>()?;
        assert_eq!(as_rgb.buffer(), &[[1u8, 2, 3]]);
        assert_eq!(
            as_rgb.buffer().as_ptr() as usize,
            data_ptr,
            "Should share the buffer"
        );

        assert_matches!(dyn_ch.try_cast::<u8>(), Err(_));
        assert_matches!(dyn_ch.try_cast::<[u8; 4]>(), Err(_));
        assert_matches!(dyn_ch.try_cast::<[u16; 3]>(), Err(_));
        Ok(())
    }

    #[test]
    fn into_vec_with_capacity_which_is_no_multiple_of_pixel_elements() -> TestResult {
        let mut data = Vec::with_capacity(4);
        data.extend_from_slice(&[[1u8, 2, 3]]);
        let dynamic = DynamicImageChannel::new_vec(data, NZ_1, NZ_1)?;
        let typed = ImageChannel::<[u8; 3]>::try_from(dynamic)?;
        assert_eq!(typed.into_vec(), vec![[1, 2, 3]]);
        Ok(())
    }

    #[test]
    fn miri_create_and_clear_vec_image_channel() -> TestResult {
        let image = ImageChannel::<u8>::new_vec(vec![0u8, 64, 128, 192], NZ_2, NZ_2)?;
        assert_eq!(image.buffer(), &[0u8, 64, 128, 192]);
        Ok(())
    }

    #[test]
    fn miri_to_vec_reuses_pointer() -> TestResult {
        let raw = vec![0u8, 64, 128, 192];
        let pointer = raw[..].as_ptr();
        let image = ImageChannel::<u8>::new_vec(raw, NZ_2, NZ_2)?;
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
        Ok(())
    }

    #[test]
    fn miri_make_mut_reuses_arc_pointer() -> TestResult {
        let raw = Arc::<[u8]>::from([0u8, 64, 128, 192].as_slice());
        let pointer = raw[..].as_ptr();
        let mut image = ImageChannel::<u8>::new_arc(raw, NZ_2, NZ_2)?;
        let ptr_mut = image.make_mut();

        assert_eq!(
            ptr_mut[..].as_ptr(),
            pointer,
            "Should reuse the buffer if it was created by arc"
        );
        Ok(())
    }

    #[test]
    fn miri_make_mut_doesnt_reuse_arc_pointer_if_not_unique() -> TestResult {
        let raw = Arc::<[u8]>::from([0u8, 64, 128, 192].as_slice());
        let _raw2 = raw.clone();
        let pointer = raw[..].as_ptr();
        let mut image = ImageChannel::<u8>::new_arc(raw, NZ_2, NZ_2)?;
        let ptr_mut = image.make_mut();

        assert_ne!(
            ptr_mut[..].as_ptr(),
            pointer,
            "Should not reuse the buffer if arc is not unique"
        );
        Ok(())
    }

    #[test]
    fn miri_clone_arc_backed_shares_memory() -> TestResult {
        let raw = Arc::<[u8]>::from([0u8, 64, 128, 192].as_slice());
        let pointer = raw[..].as_ptr();
        let image = ImageChannel::<u8>::new_arc(raw, NZ_2, NZ_2)?;
        let image2 = image.clone();

        assert_eq!(
            image2.buffer().as_ptr(),
            pointer,
            "Should share the buffer if it was created by arc"
        );
        Ok(())
    }

    #[test]
    fn miri_clone_from_vec() -> TestResult {
        let raw = vec![0u8, 64, 128, 192];
        let image = ImageChannel::<u8>::new_vec(raw, NZ_2, NZ_2)?;
        let image2 = image.clone();
        let to_vec = image.into_vec();
        let to_vec2 = image2.into_vec();

        assert_ne!(
            to_vec[..].as_ptr(),
            to_vec2[..].as_ptr(),
            "Should not share the buffer if it was created by vec"
        );
        Ok(())
    }

    #[test]
    fn miri_test_shared_arc_u16_channel() -> TestResult {
        let arc: Arc<[u16]> = vec![1].into();
        test_entire_vtable(ImageChannel::<u16>::new_arc(arc, NZ_1, NZ_1)?);
        Ok(())
    }

    #[test]
    fn miri_test_exclusive_arc_u16_channel() -> TestResult {
        test_entire_vtable(ImageChannel::<u16>::new_arc(vec![1u16].into(), NZ_1, NZ_1)?);
        Ok(())
    }

    #[test]
    fn miri_test_vec_u16_channel() -> TestResult {
        test_entire_vtable(ImageChannel::<u16>::new_vec(vec![1u16], NZ_1, NZ_1)?);
        Ok(())
    }

    #[test]
    fn miri_test_vec_rgb16_channel() -> TestResult {
        let ch = ImageChannel::<[u16; 3]>::new_vec(vec![[1u16, 2, 3]], NZ_1, NZ_1)?;
        test_entire_vtable(ch);
        Ok(())
    }

    fn test_entire_vtable<TP: PixelType>(mut channel: ImageChannelView<'_, TP>)
    where
        TP::Primitive: Default + Eq + Debug,
    {
        channel.make_mut_flat()[0] = TP::Primitive::default();
        let clone = channel.clone();
        assert_eq!(channel.make_mut_flat()[0], TP::Primitive::default());
        channel.make_mut_flat()[0] = TP::Primitive::default();

        assert_eq!(channel, clone);
    }

    #[test]
    fn miri_test_buffer_flat_bytes() -> TestResult {
        let image = ImageChannel::new_vec(vec![42u16], NZ_1, NZ_1)?;
        assert_eq!(image.buffer_flat_bytes(), &[42u8, 0]);
        Ok(())
    }

    #[test]
    fn miri_slice_channel_shares_buffer_until_make_mut() -> TestResult {
        let data = [1u16, 2];
        let mut channel = ImageChannelView::new_slice(&data, NZ_2, NZ_1)?;
        let clone = channel.clone();
        assert_eq!(clone.buffer().as_ptr(), data.as_ptr(), "Clones share");

        channel.make_mut()[0] = 3;
        assert_ne!(channel.buffer().as_ptr(), data.as_ptr());
        assert_eq!(channel.buffer(), &[3, 2]);
        assert_eq!(clone.buffer(), &[1, 2]);
        assert_eq!(clone.into_vec(), vec![1, 2]);
        assert_eq!(channel.into_vec(), vec![3, 2]);
        Ok(())
    }

    #[test]
    fn miri_test_slice_channel_vtable() -> TestResult {
        test_entire_vtable(ImageChannelView::new_slice(&[[1u16, 2]], NZ_1, NZ_1)?);
        let mut data = [[1u16, 2]];
        let ch = ImageChannelView::new_slice_mut(&mut data, NZ_1, NZ_1)?;
        test_entire_vtable(ch);
        Ok(())
    }

    #[test]
    fn miri_test_view_mut_vtable() -> TestResult {
        let arc: Arc<[u16]> = vec![1].into();
        let mut shared = ImageChannel::<u16>::new_arc(arc.clone(), NZ_1, NZ_1)?;
        test_entire_vtable(shared.view_mut());
        assert_eq!(shared.buffer(), &[0]);
        assert_eq!(&*arc, &[1]);

        let mut owned = ImageChannel::<u16>::new_vec(vec![1], NZ_1, NZ_1)?;
        test_entire_vtable(owned.view_mut().view_mut());
        assert_eq!(owned.buffer(), &[0]);
        Ok(())
    }

    #[test]
    fn miri_view_mut_of_shared_vec_channel() -> TestResult {
        let mut image = Image::<u8, 2>::new_vec(vec![1, 2], NZ_1, NZ_1)?;
        let clone = image.clone();
        let [_, mut second] = image.view_mut().into_channels();
        second.make_mut()[0] = 3;
        drop(second);
        assert_eq!(image.buffers(), [[1], [3]]);
        assert_eq!(clone.buffers(), [[1], [2]]);
        Ok(())
    }

    #[test]
    fn view_into_dynamic_channel_and_back() -> TestResult {
        let mut data = [1u8, 2];
        let view = ImageChannelView::new_slice_mut(&mut data, NZ_2, NZ_1)?;
        let mut dynamic = DynamicImageChannelView::from(view);
        dynamic.make_mut_flat::<u8>()?[1] = 3;
        let typed = ImageChannelView::<u8>::try_from(dynamic)?;
        assert_eq!(typed.buffer(), &[1, 3]);
        drop(typed);
        assert_eq!(data, [1, 3]);
        Ok(())
    }
}
