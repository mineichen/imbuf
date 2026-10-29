# 0.4.0

- Breaking: Converting a `DynamicImageView` into an `ImageView<T, CHANNELS>` (owned, `&` and `&mut`) fails, if the number of channels isn't exactly `CHANNELS`. Previously, additional channels were silently dropped (e.g. RGB from RGBA). Select the channels explicitly before the conversion instead.
- Breaking: Rename the error of owned image conversions from `IncompatibleImageError<TInput>` to `IncompatibleImage<TInput>`. It hands back the input in `image` and the cause in `error`, which is the new `IncompatibleImageError` enum (the name now refers to the cause). Owned channels return the new `IncompatibleChannel<TInput>` with the input in `channel` and the cause (`IncompatibleChannelError`) in `error`. Both don't implement `Error` anymore (the input doesn't have to be `Debug`): `into_error` returns the cause as `impl Error + Send + Sync + 'static` (e.g. for `map_err`), `Display` prints it (so `?` works for anything accepting `Display`, like `testresult`). Conversions of borrowed images/channels return the cause directly.
  - `TryFrom<DynamicImageView> for ImageView` and `TryFrom<[ImageChannelView; N]> for ImageView` fail with `IncompatibleImage<_>`.
  - `TryFrom<DynamicImageChannelView> for ImageChannelView` fails with `IncompatibleChannel<DynamicImageChannelView>` instead of the plain `DynamicImageChannel`.
  - `TryFrom<&DynamicImageView>`/`&mut` for `ImageView` fail with `IncompatibleImageError` instead of `IncompatibleImageError<&DynamicImage>`, `TryFrom<&DynamicImageChannelView>`/`&mut` for `&ImageChannelView` with `IncompatibleChannelError` instead of `()`.
  - Add the public, `#[non_exhaustive]` error types `IncompatibleImageError` (`Channel`, `MixedImageSizes`, `ChannelCount`), `IncompatibleChannelCount` (`expected`, `actual`), `IncompatibleChannelError` (`actual` and optional `required` `PixelLayout`, `aligned`, `target_pixel_type`) and `PixelLayout` (`pixel_elements`, `bytes_per_primitive`). `IncompatibleChannelError::new` and `PixelLayout::new` let conversions in other crates (e.g. `imbuf-image`) report them. `DynamicImageChannelView::pixel_layout` returns the `PixelLayout` of a channel.
  - Add `DynamicImageView::ensure_single_channel`, which fails with `IncompatibleChannelCount` for planar images (e.g. for libraries, which only support interleaved images).
- Breaking: Channels and images either own their buffers or borrow them for a lifetime `'a`. Borrowed images can be created ad-hoc (e.g. from a slice of another library) without an owned `ImageChannel` to refer to.
  - Rename `ImageChannels<[T; N]>` to `Image<T, N>` (owned) and add `ImageView<'a, T, N>` (borrowed), which always contain `N` channels. The same applies to `ImageChannel<TP>`/`ImageChannelView<'a, TP>`, `DynamicImageChannel`/`DynamicImageChannelView<'a>` and `DynamicImage`/`DynamicImageView<'a>`.
  - Owned and borrowed types are aliases of the generic `ImageBase<L, T, N>`, `ImageChannelBase<L, TP>`, `DynamicImageBase<L>` and `DynamicImageChannelBase<L>`, where `L` is `Owned` or `Borrowed<'a>` (sealed trait `BufferLifetime`), so code can be generic over owned and borrowed images. Owned types convert into borrowed types of any lifetime with `From` (e.g. `ImageView::from(image)`), `'static` borrowed types into owned types. They compare equal across owned and borrowed (`PartialEq`).
  - Owned types have no lifetime parameter, as a lifetime parameter (even `'static`) makes futures, which hold them in stream adapters across `.await`, fail the `Send` check due to compiler bugs (rust-lang/rust#110338, rust-lang/rust#64552).
  - `Image::from_interleaved` and `Image::from_planar_image` accept owned and borrowed images.
  - Remove `ImageRef`, `ImageMut` and `BorrowableImageChannel` with ImageView. `TryFrom<&mut DynamicImage> for ImageView` and `view_mut` create borrowed images, which write into the borrowed buffers.
  - Add `ImageChannelView::new_slice`/`new_slice_mut`, `view`/`view_mut` on channels and the same methods on images. Shared borrows copy the buffer on mutable access, exclusive borrows write into the borrowed buffer and copy it on `clone`.
  - The functions of `ImageChannelVTable` receive a `DynamicImageChannelView<'a>` and have to be generic over `'a` (e.g. `unsafe extern "C-unwind" fn clone<'a>(&DynamicImageChannelView<'a>) -> DynamicImageChannelView<'a>`). They are called for owned channels too. Channels returned by `clone` or assigned in `make_mut` have to stay valid as long as the input channel would.
  - `from_raw_parts` and `new_with_vtable` create owned or borrowed channels, whose buffer has to stay valid as long as the channel.
- Breaking: change `from_flat_interleaved` to work with two params width and height instead of a tuple of the two
- Breaking: `DynamicImageChannel` is a struct instead of an enum (`U8`, `U16`, `F32`). It only knows the size of its primitive (`DynamicImageChannel::bytes_per_primitive`), so its buffer can be viewed as any `bytemuck::Pod` primitive with the same size and alignment.
  - `DynamicImageChannel::buffer_flat::<T>` and `DynamicImageChannel::make_mut_flat::<T>` fail with `IncompatibleChannelError`, if the size or alignment of `T` doesn't match.
  - Add `DynamicImageChannel::buffer::<T>` (fails with `IncompatibleChannelError`, if `pixel_elements`, the size or the alignment don't match) and `unsafe DynamicImageChannel::buffer_unchecked::<T>`.
  - Add `DynamicImageChannel::try_cast`/`try_cast_mut`, which view the channel as `&ImageChannel<TP>`/`&mut ImageChannel<TP>` and fail with `IncompatibleChannelError`. They replace `ImageChannel<DynamicSize<_>>::try_cast`, which returned an `Option`.
  - `try_cast`, `try_cast_mut` and `TryFrom<DynamicImageChannel> for ImageChannel<TP>` (incl. `&`/`&mut`) check `pixel_elements`, the primitive size and the alignment.
  - Add `unsafe DynamicImageChannel::buffer_flat_unchecked::<T>` for callers which know the primitive.
  - `PartialEq` compares the bytes of the buffers (e.g. `NaN == NaN`, but `0.0 != -0.0` for `f32` channels).
  - `DynamicImage::is_uniform` compares the primitive size instead of the primitive type.
- Breaking: Remove `DynamicSize`, `RuntimePixelType` and `ImageChannel<DynamicSize<_>>`. Use `DynamicImageChannel` instead: `DynamicImageChannel::new_vec` takes a `Vec` of typed pixels (e.g. `Vec<[u8; 3]>`) instead of a flat `Vec` and `pixel_elements` like `ImageChannel::new_vec_dynamic`. For flat buffers, use `Image::new_vec_flat` and convert it, or the new `unsafe DynamicImageChannel::from_vec_flat_unchecked`/`from_arc_flat_unchecked`.
- Breaking: Rename `Image::buffer_flat_mut` to `make_mut_flat` and `ImageChannel::primitive_make_mut` to `make_mut_flat` to rise awareness of allocation and use same terminology
- Breaking: Conversions between `ImageChannel<TP>`/`Image<TP, N>` and `DynamicImageChannel`/`DynamicImage` require `TP::Primitive: bytemuck::Pod`.
- Breaking: `PixelTypePrimitive` is no longer sealed and has no methods, so custom primitives can be used. It requires `Copy` instead of `Clone`. Primitives which are not `Pod` (e.g. a wrapper around `NonZeroU8`) work with `Image` and `ImageChannel`, but can't be converted into a `DynamicImage`. Zero sized primitives are rejected at compile time.
- Breaking: `PixelType` is sealed (primitives and arrays of primitives) and contains `Primitive` directly.
- Breaking: `ImageChannelVTable` isn't generic over the primitive anymore. Its functions receive a `DynamicImageChannelView`, whose buffer is a `*const u8` (see `DynamicImageChannelView::raw_ptr`).
  - The vtable functions use the `"C-unwind"` ABI, so panics propagate instead of aborting.
  - Add `ImageChannelVTable::bytes_per_primitive`.
- Breaking: Remove `UnsafeImageChannel`; add `unsafe DynamicImageChannelView::from_raw_parts`, which documents the invariants of a channel, and `raw_ptr`/`raw_data`/`raw_vtable` to access the raw parts, e.g. when implementing a vtable. `make_mut` implementations mutate a channel by assigning a new one (the old channel is dropped via its vtable).
  - Remove `UnsafeImageChannel::new_vec` and `UnsafeImageChannel::new_arc`. Use `DynamicImageChannel::new_vec` or `ImageChannel::new_vec`/`new_arc` instead.
- Add `DynamicImageChannelView::ensure_valid_flat_len(width, height, pixel_elements, buffer_len) -> Result<(), InvalidLenError>`, which checks that `width * height * pixel_elements` matches `buffer_len` and doesn't overflow `usize` (`InvalidLenError`: `Overflow` or `BufferMismatch { expected, actual }`), and `ensure_valid_len(width, height, buffer_len)` beside it, which checks `width * height` for buffers, which store one element per pixel. Use them to validate untrusted dimensions, e.g. before `DynamicImageChannelView::from_raw_parts`. In debug builds, `from_raw_parts` asserts that the calculation of the buffer size doesn't overflow. For images, `ImageView::ensure_valid_len(width, height, len)` checks `width * height * CHANNELS` (e.g. for `Image::new_vec`) and `ImageView::ensure_valid_flat_len(width, height, len)` checks `width * height * T::ELEMENTS * CHANNELS` (e.g. for `Image::new_vec_flat`).
- Breaking: Safe constructors return `Result<_, InvalidLenError>` instead of panicking, if the buffer length doesn't match the dimensions or their multiplication overflows `usize`. This affects `ImageChannel::new_vec`/`new_arc`, `ImageChannelView::new_slice`/`new_slice_mut`, `DynamicImageChannel::new_vec`, `Image::new_vec`/`new_vec_flat`/`new_arc`/`from_flat_interleaved`/`from_planar` and `ImageView::new_slice`/`new_slice_mut`. `expected` and `actual` count the elements of the input (e.g. primitives for `Image::new_vec_flat`). Constructors from already valid images (`from_interleaved`, `from_planar_image`, `view`, `view_mut`) stay infallible. Unsafe constructors (`DynamicImageChannelView::from_raw_parts`, `ImageChannelView::new_with_vtable`, `DynamicImageChannel::from_vec_flat_unchecked`/`from_arc_flat_unchecked`) only check their invariants with `debug_assert!`.
- Fix: All constructors fail on overflowing dimension multiplications or sums, instead of comparing the buffer size against a wrapped value (in release builds, a too small buffer could be accepted, so the channel read out of bounds).
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
