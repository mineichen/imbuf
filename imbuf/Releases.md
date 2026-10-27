# 0.4.0

- Breaking: Channels and images have a lifetime `'a` for borrowed buffers, which is `'static` for owned buffers. Borrowed images can be created ad-hoc (e.g. from a slice of another library) without an owned `ImageChannel` to refer to.
  - Rename `ImageChannels<[T; N]>` to `ImageView<'a, T, N>`, which always contains `[ImageChannelView<'a, T>; N]`. `Image<T, N>` is an alias of `ImageView<'static, T, N>`.
  - `ImageChannel<TP>` is an alias of `ImageChannelView<'static, TP>`, `DynamicImageChannel` of `DynamicImageChannelView<'static>` and `DynamicImage` of `DynamicImageView<'static>`.
  - Remove `ImageRef`, `ImageMut` and `BorrowableImageChannel` with ImageView. `TryFrom<&mut DynamicImageView> for ImageView` and `ImageView::view_mut` create borrowed images, which write into the borrowed buffers.
  - Add `ImageChannelView::new_slice`/`new_slice_mut`, `ImageChannelView::view`/`view_mut` and the same methods on `ImageView`. Shared borrows copy the buffer on mutable access, exclusive borrows write into the borrowed buffer and copy it on `clone`.
  - The functions of `ImageChannelVTable` receive a `DynamicImageChannelView<'a>` and have to be generic over `'a` (e.g. `unsafe extern "C-unwind" fn clone<'a>(&DynamicImageChannelView<'a>) -> DynamicImageChannelView<'a>`).
  - `DynamicImageChannelView::from_unsafe` and `ImageChannelView::new_with_vtable` create channels of any lifetime, whose buffer has to stay valid for `'a`.
- Breaking: change `from_flat_interleaved` to work with two params width and height instead of a tuple of the two
- Breaking: `DynamicImageChannel` is a struct instead of an enum (`U8`, `U16`, `F32`). It only knows the size of its primitive (`DynamicImageChannel::primitive_size`), so its buffer can be viewed as any `bytemuck::Pod` primitive with the same size and alignment.
    - `DynamicImageChannel::buffer_flat::<T>` and `DynamicImageChannel::make_mut_flat::<T>` return `None` if the size or alignment of `T` doesn't match.
  - `DynamicImageChannel::try_cast`, `DynamicImageChannel::try_cast_mut` and `TryFrom<DynamicImageChannel> for ImageChannel<TP>` (incl. `&`/`&mut`) check `pixel_elements`, the primitive size and the alignment.
  - Add `unsafe DynamicImageChannel::buffer_flat_unchecked::<T>` for callers which know the primitive.
  - `PartialEq` compares the bytes of the buffers (e.g. `NaN == NaN`, but `0.0 != -0.0` for `f32` channels).
  - `DynamicImage::is_uniform` compares the primitive size instead of the primitive type.
- Breaking: Remove `DynamicSize`, `RuntimePixelType` and `ImageChannel<DynamicSize<_>>`. Use `DynamicImageChannel` instead (e.g. `DynamicImageChannel::new_vec` instead of `ImageChannel::new_vec_dynamic`).
- Breaking: Rename `Image::buffer_flat_mut` to `make_mut_flat` and `ImageChannel::primitive_make_mut` to `make_mut_flat`  to rise awareness of allocation and use same terminology
- Breaking: Conversions between `ImageChannel<TP>`/`Image<TP, N>` and `DynamicImageChannel`/`DynamicImage` require `TP::Primitive: bytemuck::Pod`.
- Breaking: `PixelTypePrimitive` is no longer sealed and has no methods, so custom primitives can be used. Primitives which are not `Pod` (e.g. a wrapper around `NonZeroU8`) work with `Image` and `ImageChannel`, but can't be converted into a `DynamicImage`. Zero sized primitives are rejected at compile time.
- Breaking: `PixelType` is sealed (primitives and arrays of primitives) and contains `Primitive` directly.
- Breaking: `UnsafeImageChannel` and `ImageChannelVTable` aren't generic anymore (`ptr: *const u8`).
  - The vtable functions receive a `DynamicImageChannel` and use the `"C-unwind"` ABI, so panics propagate instead of aborting.
  - Add `ImageChannelVTable::primitive_size`.
- Breaking: `UnsafeImageChannel` doesn't implement `Drop` anymore (dropping it leaks), `DynamicImageChannel` drops the channel instead. Before, safe code could create an `UnsafeImageChannel` whose `Drop` called the vtable with arbitrary fields.
  - Add `unsafe DynamicImageChannel::from_unsafe`, which documents the invariants of a channel.
  - Add `DynamicImageChannel::get_unsafe` and `unsafe DynamicImageChannel::get_unsafe_mut` to access the raw parts, e.g. when implementing a vtable.
  - Remove `UnsafeImageChannel::new_vec` and `UnsafeImageChannel::new_arc`. Use `DynamicImageChannel::new_vec` or `ImageChannel::new_vec`/`new_arc` instead.
- Breaking: `ImageChannel::buffer`, `ImageChannel::buffer_flat` and `ImageChannel::buffer_flat_bytes` are no longer `const`. `ImageChannel::buffer_flat_bytes` requires `TP::Primitive: bytemuck::NoUninit`.
- Breaking: Remove the deprecated `external` module and the `image_0_25` feature. Use the `imbuf-image` crate instead.
- Breaking: Remove deprecated `RgbImageInterleaved`, `LumaImage`, `RgbaImageInterleaved`, `RgbImagePlanar` and `RgbaImagePlanar`
- Allow more pixel primitives from std (Support for i8, i16, u32, i32, u64, i64, usize, isize, f64)
- Fix: `TryFrom<&DynamicImageChannel> for &ImageChannel<TP>` (and `&mut`) bind the lifetime of the result to the input. Before, any lifetime could be returned.
- Fix: `Image::new_vec_flat` and `ImageChannel::into_vec` deallocated with a wrong layout if the capacity wasn't a multiple of the pixel elements.
- Fix: `make_mut` on a shared channel of a multi channel `Image` with more than one pixel element panicked.

# 0.3.4

- Deprecate the `external` module (`image_0_25` integration). It now lives in the separate `imbuf-image` crate, which can be depended on directly.
  - The old API keeps working during migration.
  - `imbuf_image::ImbufAsImage::as_rust_image_dynamic` and `ImbufAsImageMut::as_rust_image_dynamic_mut` borrow an `imbuf` buffer as a `DynamicRefImage` / `DynamicRefImageMut` without copying.
- Add `Image<_, N>::first` / `first_mut` to borrow the single `ImageChannel` of all images (checked at compile time in all constructors).

# 0.3.3

- Add `ImageChannel<DynamicSize<_>>::new_vec_dynamic`

# 0.3.2

- Fix build with the `image_0_25` feature

# 0.3.1

- Impement `TryFrom<ImageChannel<DynamicSize<T>> for ImageChannel<T::Primitive>`
- Implement `Image<T, 1>::from_channel(ch: ImageChannel<T>) -> Self`

# 0.3.0

- Allow trying to cast `DynamicImageChannel<DynamicSize<T>>` to `DynamicImageChannel<T>`
- Breaking: Remove `From<&DynamicImage> for Image<T, N>`
- Implement `From<&DynamicImage> for ImageRef<&T, N>`
- Implement `From<&mut DynamicImage> for ImageRef<&mut T, N>`
- Change `Image` to a `ImageChannels` in order to support Images with shared or mutable `ImageChannel` (beside owned, which worked before)
- Add `Image`, `ImageRef` and `ImageMut` typedefs to `ImageChannels`
- Add `image_0_25::DynamicRefImage0_25`, which implements `TryFrom<Image<T,N>>`
- Implement `DynamicRefImage0_25::write_to` to encode a image
- Add `DynamicImage::len_nonzero`, which returns a `NonZeroUsize`
- Add `DynamicImage::is_uniform`, which returns true if all dimensions (width, height, pixel_elements) and datatype are equal
- Add `DynamicImage::mapped_channels`,`DynamicImage::mapped_channels_mut` and `DynamicImage::into_mapped_channels` to get channels without unwrapping the first from a iterator
- Add Into-Impls for all steps between `ImageChannel` -> `ImageChannel<DynamicSize<_>>` -> `DynamicImageChannel`

# 0.2.0

- Introduce ImageChannel, which represents 1 channel in a Image
- Add vtable per `ImageChannel` instead of per Image (Remove UnsafeImage, ImageVtable)
  - Move pixel size to the ImageChannelVTable instead of having it only in the typesystem.
- rename `Image::flat_buffer` to `Image::buffer_flat`
- Require T of `Image<T>` to be of type `PixelType` (only required `'static` before)

# 0.1.0

- Initial release with one vtable per image
- Support for DynamicImage
