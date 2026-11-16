#![doc = include_str!("../README.md")]

use std::num::NonZeroU32;

use imbuf::Image;
use js_sys::{Array, Promise, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Blob, ColorSpaceConversion, ImageBitmap, ImageBitmapOptions, OffscreenCanvas,
    OffscreenCanvasRenderingContext2d, PremultiplyAlpha, Window, WorkerGlobalScope,
};

/// Decodes an image from `data` into an RGBA [`imbuf::Image`] with the image decoders of the
/// browser.
///
/// Every format the browser can decode is supported (e.g. PNG, JPEG, GIF, WebP, AVIF, BMP). The
/// browser converts every image to 8 bit RGBA, so e.g. grayscale or 16 bit images become 8 bit
/// RGBA too. No browser API decodes to grayscale: WebCodecs' `ImageDecoder` only produces YUV or
/// 8 bit RGB(A) frames. The pixels are not premultiplied by alpha and embedded color profiles
/// are ignored.
///
/// Works in the main thread (`Window`) and in workers (`WorkerGlobalScope`). Requires
/// `createImageBitmap` and `OffscreenCanvas`.
///
/// # Errors
/// The [`JsValue`] thrown by the browser (e.g. a `DOMException`, if `data` can't be decoded) or a
/// JavaScript `Error`, if neither `Window` nor `WorkerGlobalScope` is available or the decoded
/// image has no `imbuf` representation (e.g. a zero dimension).
pub async fn load_from_memory(data: &[u8]) -> Result<Image<[u8; 4], 1>, JsValue> {
    let bitmap: ImageBitmap = JsFuture::from(create_image_bitmap(data)?)
        .await?
        .dyn_into()?;
    let result = bitmap_to_image(&bitmap);
    // Releases the decoded pixels without waiting for the garbage collector
    bitmap.close();
    result
}

fn create_image_bitmap(data: &[u8]) -> Result<Promise, JsValue> {
    // Copies `data` into a JavaScript buffer, so it doesn't have to outlive the decoding
    let blob = Blob::new_with_u8_array_sequence(&Array::of1(&Uint8Array::from(data)))?;
    let options = ImageBitmapOptions::new();
    // Keep the samples as they are encoded
    options.set_premultiply_alpha(PremultiplyAlpha::None);
    options.set_color_space_conversion(ColorSpaceConversion::None);

    let global = js_sys::global();
    if let Some(window) = global.dyn_ref::<Window>() {
        window.create_image_bitmap_with_blob_and_image_bitmap_options(&blob, &options)
    } else if let Some(worker) = global.dyn_ref::<WorkerGlobalScope>() {
        worker.create_image_bitmap_with_blob_and_image_bitmap_options(&blob, &options)
    } else {
        Err(js_error(
            "createImageBitmap requires a Window or a WorkerGlobalScope",
        ))
    }
}

/// Reads the RGBA pixels of `bitmap` by drawing it onto an [`OffscreenCanvas`]
fn bitmap_to_image(bitmap: &ImageBitmap) -> Result<Image<[u8; 4], 1>, JsValue> {
    let (width, height) = (bitmap.width(), bitmap.height());
    let (Some(nonzero_width), Some(nonzero_height)) =
        (NonZeroU32::new(width), NonZeroU32::new(height))
    else {
        return Err(js_error("Neither width nor height can be 0"));
    };

    let canvas = OffscreenCanvas::new(width, height)?;
    let context: OffscreenCanvasRenderingContext2d = canvas
        .get_context("2d")?
        .ok_or_else(|| js_error("OffscreenCanvas has no 2d context"))?
        .dyn_into()?;
    context.draw_image_with_image_bitmap(bitmap, 0., 0.)?;
    let pixels = context
        .get_image_data(0., 0., f64::from(width), f64::from(height))?
        .data()
        .0;

    Image::new_vec_flat(pixels, nonzero_width, nonzero_height)
        .map_err(|error| js_error(&error.to_string()))
}

fn js_error(message: &str) -> JsValue {
    js_sys::Error::new(message).into()
}
