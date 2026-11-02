# Soundness review (imbuf, imbuf-image, imbuf-opencv)

## Context

The user asked for a soundness review of the whole imbuf workspace (`imbuf`, `imbuf-image`, `imbuf-opencv`), with explanations before any fix. Scope covered every `unsafe` block, the vtable contract, the storages (Vec, Arc, SharedVec, Slice, SliceMut, OpenCV Mat), the owned/borrowed refactor (`cast_lifetime`, `as_view_ref`/`as_view_mut`, `From` conversions), and Send/Sync. `imbuf-image` contains no `unsafe`. Findings 1, 2, 4 and 5 already exist in `HEAD`; they are not caused by the refactor.

## Findings (most severe first)

### 1. UB: `Image::<[T; N], 1>::from_planar` reads uninitialized memory (high)
`imbuf/src/image.rs:433-488` (and `from_planar_image`, line 423, which forwards its `CHANNELS`).
- `CHANNELS` and `PIXEL_ELEMENTS` are independent const generics. Only `CHANNELS == 1` is special-cased.
- The loop `zip`s the channel iterators with `value.iter_mut()`. With `1 < CHANNELS < PIXEL_ELEMENTS`, the last `PIXEL_ELEMENTS - CHANNELS` entries of `value` are never written, and `value.map(|x| unsafe { x.assume_init() })` produces uninitialized `T`s. That is immediate UB from safe code.
- Example: `Image::<[u8; 3], 1>::from_planar([&a[..], &b[..]], w, h)`, or `Image::<[u8; 3], 1>::from_planar_image(&Image::<u8, 2>)`.
- With `CHANNELS > PIXEL_ELEMENTS`, surplus channels are silently dropped (not UB, but surely unintended).
- **Fix:** add `const { assert!(CHANNELS == 1 || CHANNELS == PIXEL_ELEMENTS) }` (compile error for wrong combinations). Rewrite the interleaving loop in safe code (`std::array::from_fn(|c| channels[c][i])` per pixel, or iterators), which removes both the `MaybeUninit` and the `transmute` (finding 4).

### 2. UB: alignment is only checked before the vtable may reallocate (medium)
`imbuf/src/dynamic.rs:510-525` (`make_mut_flat` / `make_mut_flat_unchecked`), `imbuf/src/channel.rs:68-77` (typed `Clone`) and `:344` (typed `make_mut_flat`).
- A dynamic channel can be viewed as any `Pod` primitive of the same size (`try_cast(_mut)`, `TryFrom`, `make_mut_flat::<T>`). `ensure_fits_primitive` checks `T`'s alignment against the *current* pointer only.
- `make_exclusive` and `clone` can replace the buffer (Arc copy, `copy_into_arc`, SharedVec `to_vec`, `clone_slice_into_arc_channel`). The new buffer is only aligned for the vtable's *own* primitive.
- After that, a typed `ImageChannelBase<_, TP>` (or the `&mut [T]` from `make_mut_flat::<T>`) may be misaligned. The re-check after reallocation is only a `debug_assert!`, so release builds produce misaligned references (UB).
- Reachable from safe code with a user-defined `Pod` primitive (e.g. `#[repr(C, align(16))]`, `#[derive(Pod)]`) of the same size as the vtable primitive but more strictly aligned. It needs an allocator or target that returns only minimally aligned memory (glibc's 16-byte malloc hides it; dlmalloc on wasm32, bump/embedded allocators don't).
- Not reachable with the built-in std primitives (same size ⇒ same alignment on supported targets).
- **Fix (recommended):** make alignment a static property of the storage.
  - Add `align_of_primitive: extern "C-unwind" fn() -> NonZeroUsize` to `ImageChannelVTable`. 0.4 is unreleased, so the FFI break is acceptable.
  - Document in `from_raw_parts`/`new_with_vtable` that `ptr` must be aligned to it.
  - `ensure_fits_primitive::<T>` additionally requires `align_of::<T>() <= vtable align`.
  - Typed views then stay valid across every reallocation, because every buffer a vtable produces is aligned for its primitive.
  - Update the internal vtables (`arc.rs`, `vec.rs`, `view.rs`, `shared_vec.rs`) and `imbuf-opencv/src/into_imbuf.rs` (`align_of::<P>()`).
  - *Alternative, no FFI change:* re-check after `make_exclusive`/`clone` and copy into a `Vec<T>` if misaligned. This is more code paths (clone, make_mut, SliceMut original) and costs runtime.

### 3. imbuf-opencv: borrowed `Mat`s from `as_opencv`/`as_opencv_mut` can escape the borrow (medium, upstream)
`imbuf-opencv/src/from_imbuf.rs` (`create_unrestricted_mat` + `BoxedRef`/`BoxedRefMut`).
- opencv-rust 0.101 doesn't implement `Deref`/`DerefMut` for `BoxedRef*`, precisely to prevent `mem::swap`.
- But the public (`#[doc(hidden)]`) safe trait method `OpenCVIntoExternContainer::opencv_into_extern_container` turns a `BoxedRef<'_, Mat>` into an owned `Mat` (`opencv/src/boxed_ref.rs:44-50`). Our Mats point at imbuf memory without a refcount (`u == null`), so the escaped `Mat` dangles once the image is dropped (use-after-free). Through `&` borrows it can even write into memory that is shared.
- **Fix:** none possible in imbuf-opencv without copying. Document it as a known opencv-rust limitation and report it upstream. imbuf-opencv's own internal uses (`planar_channel`, `clone_mat_channel`) are fine, because those Mats share the refcounted `UMatData`.

### 4. imbuf-opencv: `MatHeader` mirrors the C++ layout of `cv::Mat` (low-medium)
`imbuf-opencv/src/into_imbuf.rs:107-140` (`MatHeader`, `mat_buffer_is_exclusive`).
- Reads `u` at the field offset of OpenCV 4.x. With an OpenCV whose `cv::Mat` layout differs (e.g. 5.x), `UMatData::from_raw(garbage)` is UB.
- The `refcount` read is a plain read of a counter that other threads modify atomically. That's technically a data race, but harmless: a stale value can only cause an unnecessary copy.
- **Fix:** before trusting `u`, cross-check the mirrored fields against the API (`flags`, `dims`, `rows`, `cols`, `data`). On any mismatch, treat the buffer as non-exclusive (copy). Optionally gate it on the OpenCV major version.

### 5. `transmute::<Vec<MaybeUninit<T>>, Vec<T>>` (low)
`imbuf/src/image.rs:210` (`from_flat_interleaved`) and `:483` (`from_planar`).
- The layout of `Vec<A>` vs `Vec<B>` is unspecified, and the std docs advise `Vec::from_raw_parts` instead. It isn't miscompiled today, but it isn't guaranteed.
- **Fix:** `from_planar` is covered by the rewrite in finding 1. For `from_flat_interleaved`, use `ManuallyDrop` + `Vec::from_raw_parts` (or a safe rewrite).

### 6. imbuf-opencv `copied_channel` on unaligned external data (very low)
`into_imbuf.rs:249-262`: `buffer_flat_unchecked::<P>()` assumes `P` alignment. It's only reached when `Mat::copy`/`try_clone` fail on a Mat over user-provided unaligned memory. This is resolved by finding 2's invariant ("vtable buffers are aligned for their primitive"); add the alignment check to `dynamic_image`.

## Reviewed and sound (no change needed)
- **Refactor pieces:**
  - `cast_lifetime` (`ManuallyDrop` + field copy, no double drop).
  - `as_view_ref`, plus `as_view_mut` (unsafe, only used for vtable calls).
  - `From` owned→any view, `'static` view→owned.
  - Vtable calls with a borrow lifetime, given the documented contract: `clone`/`make_mut` results stay valid as long as the input.
  - Covariance of `Borrowed<'a>`, dropck through the generic `Drop`, sealed `BufferLifetime`.
- **SharedVec:** per-slice/total refcounts with AcqRel/Acquire (same reasoning as `Arc`). Overflow is impossible because every clone allocates.
- **Arc COW:** `Arc::get_mut` respects `Weak`; the `into_raw` pointer keeps write provenance.
- **Vec:** never shared, clone copies.
- **SliceMut `original` pointer:** an exclusive borrow for the view's lifetime. `make_original_mut` re-reads the pointer after `make_exclusive`.
- **`into_vec_flat`** detects a Vec by function pointer identity. Sound, because merged functions imply an identical allocation layout, and false negatives only copy.
- **Pixel casts:** `flat_vec`/`flat_slice`/`flat_arc` (`[P; N]` ↔ `P`, identical `ArcInner` layout); ZST primitives are rejected at compile time.
- **Send/Sync** of all storages, including Mat refcounts (atomic in OpenCV).
- **`ImageBase` equal-dimension invariant:** it can be broken via `first_mut()`, but no unsafe code depends on it. `from_planar` validates every length. Keep it that way.

## Proposed implementation order
1. Finding 1 (+5 for `from_planar`): const assert + safe interleaving. Add tests: `CHANNELS == PIXEL_ELEMENTS`, `CHANNELS == 1`, and a `compile_fail` doctest for `2 → [T; 3]`.
2. Finding 5 for `from_flat_interleaved`: `Vec::from_raw_parts`.
3. Finding 2: `align_of_primitive` in the vtable, `ensure_fits_primitive` against it, docs on `from_raw_parts`/`new_with_vtable`, all internal vtables + opencv. Test with a custom `#[repr(C, align(16))]` Pod primitive vs a vtable of a 16-byte, 4-aligned primitive: the cast must be rejected.
4. Finding 4: cross-check `MatHeader` before reading `u`.
5. Finding 3: doc note on `as_opencv`/`as_opencv_mut` + upstream issue text for the user (I won't file it myself).
6. `Releases.md` entries for the vtable field and the `from_planar` restriction.

## Verification
- `cargo test --workspace` (imbuf, imbuf-image, imbuf-opencv), including doctests and the new `compile_fail` test.
- `cargo clippy --workspace --all-targets`, `RUSTDOCFLAGS="-D warnings" cargo doc -p imbuf --no-deps`.
- Miri isn't installed here: the user runs `nix run .#miri` (covers `imbuf`, `imbuf-image`), which also exercises the new `from_planar`.
- pilatus: `cargo test --workspace` against the patched imbuf (pilatus uses `from_interleaved`/`from_planar_image`, so check that the const assert accepts its uses: `Image<u8, 3>` ↔ `[u8; 3]`).
