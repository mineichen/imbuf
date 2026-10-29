[![CI](https://github.com/mineichen/imbuf/actions/workflows/test.yml/badge.svg)](https://github.com/mineichen/imbuf/actions/workflows/test.yml)

# This crate provides flexible image buffers

- Allows transforming known [`Images`](ImageView) (e.g. RGB32F) to typeless [`DynamicImages`](DynamicImage) and back (without a single buffer copy)
- Ability to support buffers from other libraries like "opencv" without copying buffers
- Copy on write capability, so buffers can be reused if the internal representation allows it
- FFI compatible

## Interleaved vs planar images

A 2x2 RGB image can either be interleaved (memory: rgbrgbrgbrgb) or planar(memory: rrrrggggbbbb)
Interleaved RGB images are stored as `Image<[u8;3], 1>`, whereas planar images are stored as `Image<u8, 3>`.
This crate provides utilities functions to go from one representation to the other.

## Channels

[`ImageChannelView`] is a composeable building block for [`ImageView`]. If you have a special Image kind, where channels are not uniform,
feel encouraged to add your own typed Image, which implement `TryFrom<DynamicImage>` and `Into<DynamicImage>`.

## Primitives and dynamic channels

[`DynamicImageChannel`] only knows the size of its primitive (e.g. 2 bytes for `u16`). It can therefore be
viewed as any [`bytemuck::Pod`] primitive with the same size and alignment, e.g. with `ImageChannel::try_from`
or [`DynamicImageChannel::buffer_flat`]. Custom primitives can implement [`PixelTypePrimitive`]. Primitives
which aren't `Pod` (e.g. a wrapper around `NonZeroU8`) can be used in [`Image`] and [`ImageChannel`], but can't
be converted into a [`DynamicImage`].

## Typed Images

The `Image` struct represents a fully typed Image to support the most common image formats.

| Type                     | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| `Image<u8, 3>`           | RGB8 planar                                                   |
| `Image<[u8; 4], 1>`      | RGBA8 interleaved                                             |
| `Image<f32, 2>`          | LUMAA32F planar                                               |
| `ImageView<'a, u8, 1>`    | LUMA8, where buffers might be borrowed for `'a`               |
| `ImageView<'a, [u16; 3], 1>` | RGB16 interleaved, where buffers might be borrowed for `'a` |

## Owned and borrowed images

Channels and images own their buffers ([`ImageChannel`], [`Image`], [`DynamicImageChannel`],
[`DynamicImage`]) or borrow them for a lifetime `'a` ([`ImageChannelView`], [`ImageView`],
[`DynamicImageChannelView`], [`DynamicImageView`]). Borrowed images can be created ad-hoc, e.g.
from a slice of another library, without having an owned `ImageChannel` to refer to. Both share
the same API. Owned images are converted into borrowed images of any lifetime with `From`
(`ImageView::from(image)`), `'static` borrows into owned images.

Owned images have no lifetime parameter, so they can be held in `Send` futures (e.g. in stream
adapters within `tokio::spawn`). A lifetime parameter, even `'static`, triggers compiler bugs in
the `Send` check of futures ([rust-lang/rust#110338](https://github.com/rust-lang/rust/issues/110338)).

- Shared borrows (e.g. [`ImageView::new_slice`], [`ImageView::view`] or `ImageView::try_from(&dynamic)`)
  copy the buffer on mutable access (e.g. [`ImageView::make_mut`]).
- Exclusive borrows (e.g. [`ImageView::new_slice_mut`], [`ImageView::view_mut`] or
  `ImageView::try_from(&mut dynamic)`) write into the borrowed buffer. Clones copy the buffer.

```rust
use std::num::NonZeroU32;
use imbuf::{DynamicImageView, ImageView};

let mut data = [1u8, 2, 3, 4];
let two = NonZeroU32::new(2).unwrap();
{
    let mut image = ImageView::<u8, 1>::new_slice_mut(&mut data, two, two).unwrap();
    image.make_mut()[0][0] = 5;
    // Borrowed images can be converted to dynamic images and back
    let dynamic = DynamicImageView::from(image);
    let image = ImageView::<u8, 1>::try_from(&dynamic).unwrap();
    assert_eq!(image.buffer(), &[5, 2, 3, 4]);
}
assert_eq!(data, [5, 2, 3, 4]);
```

## Example which demonstrates buffer reuse

```rust
use std::{num::NonZeroU32, sync::Arc};
use imbuf::{Image, DynamicImage, ImageChannel, DynamicImageChannel};

let pixel_data = vec![42, 1, 2];
let data_addr = pixel_data.as_ptr();
let image: Image<u8, 3> = imbuf::Image::new_vec(pixel_data, NonZeroU32::MIN, NonZeroU32::MIN).unwrap();
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
let mut typed: Image<u8, 3> = dynamic_clone.try_into().unwrap();
let [r, g, b] = typed.make_mut();
r[0] = 0;
assert_eq!(data_addr, r.as_ptr(), "dynamic went out of scope, so buffer can be reused");
let expected = imbuf::Image::new_vec(vec![0, 1, 2], NonZeroU32::MIN, NonZeroU32::MIN).unwrap();
assert_eq!(expected, typed, "images with different buffer can be compared");
```
