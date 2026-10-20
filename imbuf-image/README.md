# imbuf-image

Conversions between [`imbuf`](https://crates.io/crates/imbuf) image buffers and
the [`image`](https://crates.io/crates/image) crate, without copying buffers.


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
`ImbufAsRustImage` / `ImbufAsRustImageMut` / `ImbufIntoRustImage` and
`ImageToImbuf` extension traits),
exactly like `imbuf-opencv`.
