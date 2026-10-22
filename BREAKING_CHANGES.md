# Breaking changes

Deferred breaking changes for the next breaking `imbuf` release (0.4.0).
`imbuf-image` and `imbuf-opencv` are not released yet, so changes to those
crates are tracked in `RELEASE_CHANGES.md` instead.

- Add `#[non_exhaustive]` to `DynamicImageChannel`, so new primitives (e.g.
  `f64`, `u32`, `i16`) can be added without breaking exhaustive matches
  downstream. Matches then require a wildcard arm.
- Remove the deprecated `external` module (`image_0_25` integration),
  deprecated in 0.3.4 in favor of the `imbuf-image` crate.
- (Candidate) `TryFrom<DynamicImage> for Image<T, N>` currently ignores
  channels beyond `N` (e.g. taking RGB from RGBA, see `tests/dynamic.rs`).
  Require an exact channel count instead and add an explicit method for taking
  the first `N` channels. 
- Add a method `try_extract_channel(&mut self, index: usize)` to DynamicImage, which fails 
  if it's the last channel && index=0 or if the index doesn't exist
- (Candidate) Replace the `Error = ()` of the
  `TryFrom<&DynamicImageChannel> for &ImageChannel<TP>` family with a typed
  error. Decision deferred.
- Big change:  &ImageChannels in other structures like Image are unflexible.
  Because instances have to live somewhere to refer to, this is not possible ad-hoc
  Because of this, we cannot create a Ref-Channel over e.g. a &image::DynamicImage... 
- Remove `impl DerefMut for DynamicImage` as it allows violating the never len=0 invariant
