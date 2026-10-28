# Breaking changes

Deferred breaking changes for the next breaking `imbuf` release (0.4.0).
`imbuf-image` and `imbuf-opencv` are not released yet, so changes to those
crates are tracked in `RELEASE_CHANGES.md` instead.

- (Candidate) Add an explicit method for taking the first `N` channels of a
  `DynamicImage`, as `TryFrom<DynamicImage> for Image<T, N>` requires an exact
  channel count now (e.g. taking RGB from RGBA).
- Add a method `try_extract_channel(&mut self, index: usize)` to DynamicImage, which fails 
  if it's the last channel && index=0 or if the index doesn't exist
