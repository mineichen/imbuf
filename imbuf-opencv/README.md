# imbuf-opencv

Zero-copy conversions between [`imbuf`](https://crates.io/crates/imbuf) image
buffers and `OpenCV` [`Mat`](https://docs.rs/opencv/latest/opencv/core/struct.Mat.html)
views.

## Conversion model

Conversions don't copy pixels. The lifetimes keep the buffer valid:

- `OpencvIntoImbuf` moves an owned `Mat` into an owned `imbuf::DynamicImage`,
  which holds the `Mat` (and its reference to the buffer).
- `OpencvIntoImbuf` turns a `BoxedRef<'a, Mat>` or `&'a Mat` into an image,
  which borrows the buffer for `'a` (mutable access copies it), and a
  `BoxedRefMut<'a, Mat>` or `&'a mut Mat` into an image, which writes into the
  mutably borrowed buffer (clones copy it).
- `ImbufAsOpencv` borrows an `imbuf` image as a `Mat` view, `ImbufAsOpencvMut`
  provides the same view mutably; writes through the `Mat` land in the `imbuf`
  buffer. Like `Mat::new_rows_cols_with_data`, the view is a `BoxedRef`, which
  can't outlive the borrow. Copy it (e.g. with `Mat::try_clone`) to keep the
  pixels.

A `Mat` represents one image channel, or one interleaved pixel with up to its
channel limit. Multi-channel planar `imbuf::DynamicImage` values are rejected.
Only `OpenCV` 4 is supported, as copy on write reads the reference count from the
mirrored layouts of `cv::Mat` and `cv::UMatData` (checked against their
accessors).

## Example

```rust
use std::num::NonZeroU32;

use imbuf::Image;
use imbuf_opencv::ImbufAsOpencv;
use opencv::core::MatTraitConst;

let image: Image<u8, 1> = Image::new_vec(
    vec![1, 2, 3, 4],
    NonZeroU32::new(2).expect("non-zero width"),
    NonZeroU32::new(2).expect("non-zero height"),
).unwrap();

let mat = image.as_opencv().expect("valid Mat layout");
assert_eq!(mat.rows(), 2);
assert_eq!(mat.cols(), 2);
```

## Requirements

Building this crate requires an `OpenCV` 4 installation with its development files
available to `CMake` or `pkg-config`, a C++ compiler, `CMake`, `pkg-config`, and
libclang (used by the `opencv` crate's bindings generator).

With Nix, the repository's development shell provides all of these:

```sh
nix develop
cargo test --workspace --all-features
```

## Supported versions

Only the newest supported combination of `imbuf` and `opencv` is released here.
If you need a different combination, implement the same conversions against your
own local wrapper or extension trait; see the repository README for details.

## License

MPL-2.0
