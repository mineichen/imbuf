# imbuf-tiff

Decodes TIFFs into an [`imbuf`](https://crates.io/crates/imbuf) `DynamicImage`
with their raw samples, which e.g. the `image` crate can't represent:

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

Pages which need a conversion to be meaningful (e.g. CMYK, YCbCr, palette or
1 bit) are not supported. Decode them with e.g. `imbuf-image` instead.

```rust
# fn load(data: &[u8]) -> std::io::Result<()> {
let image = imbuf_tiff::load_from_memory(data)?;
# Ok(())
# }
```
