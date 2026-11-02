# Release changes

Pending changes for the upcoming releases of the unreleased `imbuf-image` /
`imbuf-opencv` 0.1.0 crates. `imbuf` changes are listed in `imbuf/Releases.md`
(0.4.0), further breaking `imbuf` changes are tracked in `BREAKING_CHANGES.md`.

## imbuf-image 0.1.0 (unreleased)

- Implement `ImbufAsImage`, `ImbufAsImageMut` and `ImbufIntoImage` for borrowed images
  (`imbuf::ImageView<'_, _, 1>` and `imbuf::DynamicImageView<'_>`, imbuf 0.4.0).
- Split the extension traits into infallible and fallible variants, replacing
  the `type Result<'a>` "infallible-or-`Result`" GAT, which cannot be reasoned
  about in generic code and makes third-party implementations harder:
  - `ImbufAsImage` / `ImbufAsImageMut` become infallible (implemented for
    typed `Image<T, 1>` sources and `image::DynamicImage`), returning the view
    directly.
  - Fallible borrows of `imbuf::DynamicImage` stay available through the
    `TryFrom<&imbuf::DynamicImage> for DynamicRefImage<'_>` impls (and the
    `Mut` variants).
  - `ImbufIntoImage` becomes infallible for typed sources; converting an
    `imbuf::DynamicImage` (fallible, and the orphan rule forbids a `TryFrom`
    impl) moves to a separate fallible trait.
- Replace the `Result<_, String>` of `ImageIntoImbuf::into_imbuf_specific` with
  a typed error that recovers the input `imbuf::DynamicImage`.
- Make `IncompatibleLayoutError` inspectable: public fields or a
  `#[non_exhaustive]` reason enum, consistent with `IntoImbufErrorReason`.
- Remove the unreachable `IncompatibleLayoutError { channels: None,
  pixel_elements: None }` error path for `image::DynamicImage` receivers (all
  10 variants are supported).
- Add `#[non_exhaustive]` to `DynamicRefImage` and `DynamicRefImageMut`: they
  mirror `image::DynamicImage`'s variants, so they grow whenever the `image`
  crate adds a pixel format.

## imbuf-opencv 0.1.0 (unreleased)

- Implement `ImbufAsOpencv` / `ImbufAsOpencvMut` for borrowed images and channels
  (`ImageView<'_, _, 1>`, `ImageChannelView<'_, _>`, `DynamicImageView<'_>`,
  `DynamicImageChannelView<'_>`, imbuf 0.4.0).
- Construct the channels for imported `Mat`s with the public
  `DynamicImageChannelView::from_raw_parts` (imbuf 0.4.0) instead of a `transmute` from
  raw parts. Requires the `imbuf` dependency 0.4.0.
- `into_imbuf_dynamic`/`into_imbuf_specific` reject `Mat`s, whose data isn't aligned for their
  depth (only possible for `Mat`s over external data), as imbuf requires aligned buffers.
- Add `ImbufAsOpencv` / `ImbufAsOpencvMut` for `imbuf::DynamicImageChannel`.
  Its depth is derived from the primitive size (1 byte: `CV_8U`, 2 bytes:
  `CV_16U`, 4 bytes: `CV_32F`).
- Check the layout of `cv::Mat` and `cv::UMatData` before reading their mirrored fields, and
  load the reference counts atomically with acquire ordering. Only `OpenCV` 4 is supported
  (checked at compile time). If the mirrored fields don't match their accessors, it panics.
- `OpencvIntoImbuf<'a>` has the lifetime of the resulting image and is implemented for
  `BoxedRef<'a, Mat>` and `&'a Mat` (the image borrows the buffer, mutable access copies it)
  and for `BoxedRefMut<'a, Mat>` and `&'a mut Mat` (the image writes into the buffer, clones
  copy it), besides the owned `Mat`.
- `as_opencv`/`as_opencv_mut` fail with `StsOutOfRange` for images, whose width or height
  exceeds `i32::MAX`. Previously, they wrapped to negative `Mat` sizes.
- `ImbufAsOpencvMut` for typed images requires `Pod` primitives, as `OpenCV` can write any value.
