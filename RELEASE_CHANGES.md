# Release changes

Pending changes for the upcoming releases of the unreleased `imbuf-image` /
`imbuf-opencv` 0.1.0 crates. `imbuf` changes are listed in `imbuf/Releases.md`
(0.4.0), further breaking `imbuf` changes are tracked in `BREAKING_CHANGES.md`.

## imbuf-image 0.1.0 (unreleased)

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

- Construct the channels for imported `Mat`s with the public
  `DynamicImageChannel::from_unsafe` (imbuf 0.4.0) instead of a `transmute` from
  `UnsafeImageChannel`. Requires the `imbuf` dependency 0.4.0.
- Add `ImbufAsOpencv` / `ImbufAsOpencvMut` for `imbuf::DynamicImageChannel`.
  Its depth is derived from the primitive size (1 byte: `CV_8U`, 2 bytes:
  `CV_16U`, 4 bytes: `CV_32F`).
