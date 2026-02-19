# 0.3.2
- Fix build with the image_0_25 feature

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
- Add image_0_25::DynamicRefImage0_25, which implements `TryFrom<Image<T,N>>`
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
