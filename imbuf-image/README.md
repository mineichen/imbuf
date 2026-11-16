# imbuf-image

Conversions between [`imbuf`](https://crates.io/crates/imbuf) image buffers and
the [`image`](https://crates.io/crates/image) crate, with zero-copy borrows and
moves where possible.

Borrowing conversions expose the `imbuf` allocation as a `DynamicRefImage` /
`DynamicRefImageMut`. Owned imbuf-to-image conversions move uniquely owned
buffers, but copy shared buffers. Converting an owned `image::DynamicImage` into
`imbuf` moves the buffer without copying.


## Which versions are covered?

Only the **newest** supported combination of `imbuf` and `image` is released
here. Backend versions change over time, and maintaining a matrix of every
`imbuf` × `image` combination is not practical.

If you need another combination — a different `image` version, or one matching an
older `imbuf` release you are pinned to — implement the conversions yourself.
This crate is the template: see the repository `README.md` for the short recipe.

## Why is this a separate crate?

`imbuf` must not privilege any backend, so every integration lives in its own
crate. Rust's orphan rules mean this crate provides the conversions through its
own types and traits (borrowed `DynamicRefImage` / `DynamicRefImageMut` plus the
`ImbufAsImage` / `ImbufAsImageMut` / `ImbufIntoImage` and
`ImageIntoImbuf` extension traits).

## Loading images

`load_from_memory` decodes an image straight into an `imbuf::DynamicImage`,
supporting every format enabled on the `image` crate. With the `tiff` feature,
TIFFs are decoded with their raw samples, which `image` 0.25 can't represent
in its `DynamicImage`:

- Any integer (`u8`..`u64`, `i8`..`i64`) or float (`f32`, `f64`) sample type,
  e.g. a grayscale `u32` TIFF becomes `imbuf::Image<u32, 1>`.
- Up to 255 samples per pixel, e.g. a 5-band `u16` page becomes
  `imbuf::Image<[u16; 5], 1>`.
- Each page becomes a channel, so pages can differ in sample type and size,
  e.g. an `f32`, a smaller `u16` and another `f32` page.
- Previews (reduced-resolution pages such as thumbnails) are skipped,
  wherever they are in the file.
- Transparency masks are loaded like image pages, if they have a supported
  sample type. 1 bit masks are not supported (yet).
- Multi-page documents (pages flagged as "page of a multi-page document") are
  rejected.

A TIFF whose only image page needs a conversion (e.g. CMYK, palette, 1 bit) is
still decoded by `image`.
