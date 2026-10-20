# imbuf-opencv

Zero-copy conversions between [`imbuf`](https://crates.io/crates/imbuf) image
buffers and OpenCV [`Mat`](https://docs.rs/opencv/latest/opencv/core/struct.Mat.html)
views.

## Conversion model

- `ImbufAsOpencv` borrows an `imbuf` image as a `Mat` view. The view points at
  the `imbuf` allocation and cannot outlive the borrow.
- `ImbufAsOpencvMut` provides the same view mutably; writes through the `Mat`
  land in the `imbuf` buffer.
- `OpencvIntoImbuf` moves an owned `Mat` into an `imbuf` image without copying
  pixel data.

There is intentionally no conversion from an `imbuf` image to an owned `Mat`
without copying: OpenCV's allocation model cannot take ownership of an `imbuf`
allocation. Copy the borrowed view explicitly if you need an owned `Mat`.

A `Mat` represents one image channel, or one interleaved pixel with up to its
channel limit. Multi-channel planar `imbuf::DynamicImage` values are rejected.

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
);

let mat = image.as_opencv()?;
assert_eq!(mat.rows(), 2);
assert_eq!(mat.cols(), 2);
# Ok::<(), opencv::Error>(())
```

## Requirements

Building this crate requires an OpenCV 4 installation with its development files
available to CMake or `pkg-config`, a C++ compiler, CMake, `pkg-config`, and
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
