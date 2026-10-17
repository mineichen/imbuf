# Unreleased
- Deprecate the `external` module (`image_0_25` integration). It now lives in the separate `imbuf-image` crate, which can be depended on directly.
- The old API keeps working during migration.

## Migration: `imbuf::external` -> `imbuf-image`

Add the new crate (only the newest `imbuf` x `image` combination is released):

```toml
[dependencies]
imbuf-image = "0.1"
```

Types were renamed and the conversions are now provided through extension
traits, because a crate other than `imbuf` cannot write `From` / `TryFrom`
impls between two foreign types (Rust orphan rules):

| Old (deprecated in `imbuf::external`) | New (`imbuf_image`)           |
| ------------------------------------- | ----------------------------- |
| `DynamicRefImage0_25`                 | `DynamicRefImage`             |
| `IntoDynamicImage0_25Error`           | `IntoDynamicImageError`       |
| `IntoDynamicImage0_25ErrorReason`     | `IntoDynamicImageErrorReason` |
| `IntoDynamicRefImage0_25Error`        | `IntoDynamicRefImageError`    |
| `IncompatibleBufferSize`              | `IncompatibleBufferSize`      |

Conversions:

| Old                                            | New                                                   |
| ---------------------------------------------- | ----------------------------------------------------- |
| `DynamicRefImage0_25::from(&image)`            | `image.as_image_ref()` (via `ImbufToImage`)           |
| `DynamicRefImage0_25::try_from(&dynamic)`      | `dynamic.as_image_ref()` (via `ImbufToImage`)         |
| `image::DynamicImage::from(owned_image)`       | `owned_image.into_image()` (via `IntoImage`)          |
| `imbuf::DynamicImage::try_from(image_dynamic)` | `image_dynamic.into_dyn_imbuf()` (via `ImageToImbuf`) |

Remember to bring the traits into scope:

```rust
use imbuf_image::{ImbufToImage, ImageToImbuf, IntoImage};
```

If you need a different `image` version, or one matching an older `imbuf`
release, implement the same conversions yourself in your own crate. See the
repository `README.md`.

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
