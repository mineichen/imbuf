# imbuf

`imbuf` provides image buffers that abstract over their underlying storage, so
the same buffer can be shared, borrowed, or owned by different Storage implementations. 
The core idea is that **anyone can add support for any storage**. There is no
privileged set of "known" storages.

## Why integrations live in separate crates

If `imbuf` provided `Into<imbuf::Image<_, _>>` for a specific external type, we
would create a two-class system: types `imbuf` happens to know about could implement it 
and everything else could not. That would violate the core philosophy of this crate.

So the rule is:

- `imbuf` owns the buffer types and the conversion *contract*.
- Each backend lives in its own crate (`imbuf-opencv`, `imbuf-image`, ...) and
  depends on `imbuf`.
- Because of Rust's orphan rules, a backend can only add conversions through its
  own local types or traits (a wrapper and/or an extension trait)

This keeps the barrier to adding a new backend identical to the barrier the crate
itself uses.

## Which combinations are released

This repository only releases the **newest** combination of `imbuf` and each
backend. Maintaining a matrix of every `imbuf` x backend-version combination is
not practical.

## Adding support for types this repository does not provide

You do not need a dedicated crate to support a new backend or a new version. Do
exactly what the integration crates do:

1. Add your external type as a dependency (optionally behind a feature).
2. Implement the buffer traits for your storage.
3. Write the conversions through your own local types/traits.

This works for anything: an unsupported `image` version, a new image library, or
a private storage type. In particular, if you need a different `image` version,
or a backend version that matches an older (or newer) `imbuf` release, implement
the very same conversions yourself. There is no special hook to wait for: the
code in this repository is exactly the code you would write.

## Workspace layout

- `imbuf` - the core buffer crate.
- `imbuf-opencv` - OpenCV bindings.
- `imbuf-image` - `image` crate conversions.
