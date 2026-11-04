use std::{
    mem,
    num::{NonZeroU8, NonZeroUsize},
};

#[cfg(doc)]
use crate::ImageChannelView;
use crate::{PixelTypePrimitive, dynamic_channel::DynamicImageChannelView, error::PixelLayout};

/// `VTable` of the storage of a [`DynamicImageChannelView`] or [`ImageChannelView`], e.g. of the
/// buffer of another library
///
/// It's created by [`ImageChannelVTable::new`] for the primitive of the buffer. Storages with a
/// primitive type, which is only known at runtime (e.g. an `OpenCV` `Mat`, which can contain `u8`
/// or `u16`), use one vtable per primitive. `new` is a `const fn`, so the
/// `&'static ImageChannelVTable`, which channels require, can be created in a `const`, `static`
/// or inline `const` block (also in generic code):
///
/// ```
/// use std::{num::NonZeroU32, ptr};
/// use imbuf::{DynamicImageChannelView, ImageChannel, ImageChannelVTable, PixelTypePrimitive};
///
/// /// Channel, which holds its buffer in a `Box<[T]>`
/// fn boxed_channel<T: PixelTypePrimitive>(
///     buffer: Box<[T]>,
///     width: NonZeroU32,
///     height: NonZeroU32,
/// ) -> ImageChannel<T> {
///     DynamicImageChannelView::ensure_valid_len(width, height, buffer.len()).unwrap();
///     // Safety: The length is checked. The Box keeps the buffer alive until the vtable drops it
///     unsafe {
///         ImageChannel::new_with_vtable(
///             Box::into_raw(buffer).cast::<T>(),
///             width,
///             height,
///             box_vtable::<T>(),
///             ptr::null_mut(),
///         )
///     }
/// }
///
/// fn box_vtable<T: PixelTypePrimitive>() -> &'static ImageChannelVTable {
///     const {
///         &ImageChannelVTable::new::<T>(
///             clone_box::<T>,
///             make_box_mut,
///             try_make_box_mut,
///             drop_box::<T>,
///         )
///     }
/// }
///
/// unsafe extern "C-unwind" fn clone_box<'a, T: PixelTypePrimitive>(
///     channel: &DynamicImageChannelView<'a>,
/// ) -> DynamicImageChannelView<'a> {
///     // Safety: The buffer contains values of T
///     let copy: Box<[T]> = unsafe { channel.buffer_flat_unchecked::<T>() }.into();
///     // Safety: The copy has the same length and is owned by the new channel
///     unsafe {
///         DynamicImageChannelView::from_raw_parts(
///             Box::into_raw(copy).cast::<u8>(),
///             channel.width(),
///             channel.height(),
///             channel.pixel_elements(),
///             box_vtable::<T>(),
///             ptr::null_mut(),
///         )
///     }
/// }
///
/// unsafe extern "C-unwind" fn make_box_mut(_: &mut DynamicImageChannelView<'_>) {
///     // The Box is never shared, as `clone` copies it
/// }
///
/// unsafe extern "C-unwind" fn try_make_box_mut(_: &mut DynamicImageChannelView<'_>) -> bool {
///     // Exclusive without copying, so `buffer_mut` returns the buffer
///     true
/// }
///
/// unsafe extern "C-unwind" fn drop_box<T>(channel: &mut DynamicImageChannelView<'_>) {
///     let buffer = ptr::slice_from_raw_parts_mut(
///         DynamicImageChannelView::raw_ptr(channel).cast::<T>().cast_mut(),
///         channel.len_flat(),
///     );
///     // Safety: The buffer was created by `Box::into_raw` with this length
///     drop(unsafe { Box::from_raw(buffer) });
/// }
///
/// let channel = boxed_channel(vec![1u16, 2].into(), NonZeroU32::new(2).unwrap(), NonZeroU32::MIN);
/// let mut clone = channel.clone();
/// clone.make_mut()[0] = 3;
/// assert_eq!(channel.buffer(), &[1, 2]);
/// assert_eq!(clone.buffer(), &[3, 2]);
/// // Writes without copying
/// clone.buffer_mut().unwrap()[1] = 4;
/// assert_eq!(clone.buffer(), &[3, 4]);
/// ```
///
/// All functions use the "C-unwind" ABI, so panics propagate instead of aborting. They receive a
/// [`DynamicImageChannelView`], whose raw parts are accessible through the safe associated
/// functions [`DynamicImageChannelView::raw_ptr`], [`DynamicImageChannelView::raw_data`] and
/// [`DynamicImageChannelView::raw_vtable`]. To mutate a channel, replace it entirely by assigning
/// a new `DynamicImageChannelView` (the old channel is dropped via its vtable).
/// For channels of non-`Pod` primitives (only reachable through [`ImageChannelView`]), the
/// buffer is only valid for the primitive the vtable was created for.
///
/// Creating a vtable is safe, but [`DynamicImageChannelView::from_raw_parts`] and
/// [`ImageChannelView::new_with_vtable`] require its functions to fulfill the contract
/// described in [`ImageChannelVTable::new`].
///
/// # ABI
/// The vtable is `#[repr(C)]` and starts with an ABI version (currently `0`), followed by the
/// entries of that version. Later versions only append entries, which have a default for older
/// vtables (e.g. one created by another dylib, which uses an older version of this crate), so
/// entries can be added without breaking changes.
#[repr(C)]
pub struct ImageChannelVTable {
    /// Version of the layout. Entries, which were added after this version, aren't present and
    /// must not be read.
    pub(crate) abi_version: u64,
    pub(crate) clone: for<'a> unsafe extern "C-unwind" fn(
        &DynamicImageChannelView<'a>,
    ) -> DynamicImageChannelView<'a>,
    pub(crate) make_mut: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>),
    /// `make_mut` without copying, see [`ImageChannelVTable::new`]
    pub(crate) try_make_mut: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>) -> bool,
    pub(crate) drop: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>),
    /// Size of the primitive of the channel in bytes. It's the same for the entire lifetime of the
    /// channel and for channels created by `clone` and `make_mut`. It receives the channel, so
    /// storages with a runtime primitive type could use a single vtable (not public yet).
    pub(crate) primitive_size:
        unsafe extern "C-unwind" fn(&DynamicImageChannelView<'_>) -> NonZeroUsize,
    /// Alignment, which every buffer of the channel is guaranteed to have, as a power of two.
    /// It's the same for the entire lifetime of the channel, and channels created by `clone` and
    /// `make_mut` have at least the same alignment.
    pub(crate) primitive_alignment:
        unsafe extern "C-unwind" fn(&DynamicImageChannelView<'_>) -> NonZeroUsize,
    /// [`PixelTypePrimitive::TYPE_HINT`] of the primitive the vtable was created for. Its name
    /// is valid UTF-8, which `new` checks at compile time.
    pub(crate) type_hint: [u8; 8],
}

// The version has to stay the first field, so every version of the crate can read it
const _: () = assert!(mem::offset_of!(ImageChannelVTable, abi_version) == 0);

impl ImageChannelVTable {
    /// ABI version of the vtables created by this version of the crate
    const ABI_VERSION: u64 = 0;

    /// Vtable for channels, whose buffer contains values of the primitive `T`. The size,
    /// alignment and [type hint](PixelTypePrimitive::TYPE_HINT) of the primitive are derived
    /// from `T`. An invalid type hint fails to compile.
    ///
    /// The functions have to fulfill the following contract for channels created with this
    /// vtable:
    /// - They work with channels of any lifetime `'a`, as owned channels can be coerced to
    ///   shorter lifetimes. `'a` is alive during all calls.
    /// - `clone` returns a channel with the same dimensions, `pixel_elements` and content, which
    ///   can be used independently. The buffer can be shared until `make_mut` is called. The
    ///   vtable of the returned channel is created for `T` as well (or for a primitive with the
    ///   same size and at least the same alignment).
    /// - `make_mut` ensures that the buffer is exclusively owned by the channel, so it can be
    ///   mutated. The entire channel can be replaced (assigning a `DynamicImageChannelView` drops
    ///   the old one), but it has to keep the dimensions, `pixel_elements` and content, and the
    ///   primitive like `clone`.
    /// - `try_make_mut` is `make_mut` without copying the buffer, which enables
    ///   [`ImageChannelView::buffer_mut`], [`DynamicImageChannelView::buffer_mut`] and their
    ///   `flat` versions. It returns `true`, if the buffer is exclusively owned by the channel
    ///   afterwards, so it can be mutated like after `make_mut` (e.g. because a reference count
    ///   is 1). Otherwise, it returns `false`, e.g. if the buffer is shared or borrowed
    ///   immutably. Returning `false` is always sound, but then `buffer_mut` returns `None`.
    ///   Like `make_mut`, it can replace the channel, but has to keep the dimensions,
    ///   `pixel_elements`, content and primitive.
    /// - `drop` releases the buffer and `data`. The channel is not used afterwards.
    /// - `clone` can be called concurrently, as channels are `Sync`.
    ///
    /// Typed views rely on this contract, so violating it is undefined behavior. Debug builds
    /// check the dimensions, `pixel_elements` and the primitive after `clone`, `make_mut` and
    /// `try_make_mut` and panic, if they changed.
    #[must_use]
    pub const fn new<T: PixelTypePrimitive>(
        clone: for<'a> unsafe extern "C-unwind" fn(
            &DynamicImageChannelView<'a>,
        ) -> DynamicImageChannelView<'a>,
        make_mut: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>),
        try_make_mut: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>) -> bool,
        drop: unsafe extern "C-unwind" fn(&mut DynamicImageChannelView<'_>),
    ) -> Self {
        // Checks the type hint at compile time, so `DynamicImageChannelView::type_hint` can't fail
        const {
            assert!(
                crate::pixel::parse_type_hint(&T::TYPE_HINT).is_some(),
                "Invalid PixelTypePrimitive::TYPE_HINT, the bytes before the first nul byte must be a str"
            );
        };
        Self {
            abi_version: Self::ABI_VERSION,
            clone,
            make_mut,
            try_make_mut,
            drop,
            primitive_size: primitive_size::<T>,
            primitive_alignment: primitive_alignment::<T>,
            type_hint: T::TYPE_HINT,
        }
    }
}

/// `primitive_size` of [`ImageChannelVTable::new`] for the primitive `T`
unsafe extern "C-unwind" fn primitive_size<T: PixelTypePrimitive>(
    _: &DynamicImageChannelView<'_>,
) -> NonZeroUsize {
    const { PixelLayout::of_primitive::<T>(NonZeroU8::MIN).primitive_size }
}

/// `primitive_alignment` of [`ImageChannelVTable::new`] for the primitive `T`
unsafe extern "C-unwind" fn primitive_alignment<T: PixelTypePrimitive>(
    _: &DynamicImageChannelView<'_>,
) -> NonZeroUsize {
    const { PixelLayout::of_primitive::<T>(NonZeroU8::MIN).primitive_alignment }
}

// Workaround inability to have static which uses outer Generics
pub(crate) trait ChannelFactory<T> {
    const VTABLE: &'static ImageChannelVTable;
}
