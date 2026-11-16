# imbuf-websys

Decodes images into an RGBA [`imbuf`](https://crates.io/crates/imbuf) `Image<[u8; 4], 1>`
with the image decoders of the browser, so no decoder has to be compiled into
the WebAssembly binary.

`load_from_memory` supports every format the browser can decode (e.g. PNG,
JPEG, GIF, WebP, AVIF). It uses `createImageBitmap` and an `OffscreenCanvas`, so
it works in the main thread and in workers. The result is an
`imbuf::Image<[u8; 4], 1>` with non-premultiplied RGBA pixels, because browsers
convert every image to 8 bit RGBA. Errors are the `JsValue`s thrown by the
browser.

```rust,no_run
# async fn load(data: &[u8]) -> Result<(), wasm_bindgen::JsValue> {
let image = imbuf_websys::load_from_memory(data).await?;
# Ok(())
# }
```

For 16 bit, float or multi-channel images, use `imbuf-image` or `imbuf-tiff`.
