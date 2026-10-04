[![CI](https://github.com/mineichen/image-buffer/actions/workflows/test.yml/badge.svg)](https://github.com/mineichen/image-buffer/actions/workflows/test.yml)

# This crate provides flexible image buffers

- Allows transforming known [`Images`](ImageChannels) (e.g. RGB32F) to typeless [`DynamicImages`](DynamicImage) and back (without a single buffer copy)
- Ability to support buffers from other libraries like "opencv" without copying buffers
- Copy on write capability, so buffers can be reused if the internal representation allows it
- FFI compatible

## Interleaved vs planar images

A 2x2 RGB image can either be interleaved (memory: rgbrgbrgbrgb) or planar(memory: rrrrggggbbbb)
Interleaved RGB images are stored as `Image<[u8;3], 1>`, whereas planar images are stored as `Image<u8, 3>`.
This crate provides utilities functions to go from one representation to the other.

## Channels

[`ImageChannel`] is a composeable building block for [`Image`]. If you have a special Image kind, where channels are not uniform,
feel encouraged to add your own typed Image, which implement `TryFrom<DynamicImage>` and `Into<DynamicImage>`.

## Typed Images

The `Image` struct represents a fully typed Image to support the most common image formats.

| Type                   | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| `Image<u8, 3>`         | RGB8 planar                                            |
| `Image<[u8; 4], 1>`    | RGBA8 interleaved                                      |
| `Image<f32, 2>`        | LUMAA32F planar                                        |
| `ImageRef<u8, 1>`      | LUMA8, where buffers are borrowed                      |
| `ImageMut<[u16;3], 1>` | RGB16 Interleaved, where buffers are mutually borrowed |

## Example which demonstrates buffer reuse

```rust
use std::{num::NonZeroU32, sync::Arc};
use imbuf::{Image, DynamicImage, ImageChannel, DynamicImageChannel};

# fn test() -> Result<(), Box<dyn std::error::Error>> {
let pixel_data = vec![42, 1, 2];
let data_addr = pixel_data.as_ptr();
let image: Image<u8, 3> = imbuf::Image::new_vec(pixel_data, NonZeroU32::MIN, NonZeroU32::MIN);
let dynamic = DynamicImage::from(image);
let dynamic_clone = dynamic.clone();
{
    assert_eq!(1, dynamic.last().width().get(), "DynamicImage alwas contains >=1 channels");
    let untyped_channel = dynamic.into_iter().next().unwrap();
    let mut typed_channel = ImageChannel::<u8>::try_from(untyped_channel).unwrap();
    assert_eq!(42, typed_channel.buffer()[0], "Value of the first channel is 0");
    assert_eq!(data_addr, typed_channel.buffer().as_ptr(), "shared pointer reuses the buffer");
    assert_ne!(data_addr, typed_channel.make_mut().as_ptr(), "dynamic_clone prevents mut buffer reuse");
}
let mut typed: Image<u8, 3> = dynamic_clone.try_into()?;
let [r, g, b] = typed.make_mut();
r[0] = 0;
assert_eq!(data_addr, r.as_ptr(), "dynamic went out of scope, so buffer can be reused");
let expected = imbuf::Image::new_vec(vec![0, 1, 2], NonZeroU32::MIN, NonZeroU32::MIN);
assert_eq!(expected, typed, "images with different buffer can be compared");

# Ok(())
# }
```
