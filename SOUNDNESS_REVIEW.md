# Soundness review (imbuf, imbuf-image, imbuf-opencv)

Open soundness issues of the workspace, most severe first.

## 1. `SharedVec` buffer is taken as a `Vec` of another primitive (very low, theoretical)
`imbuf/src/shared_vec.rs` (`try_into_vec`, used by the in-place `transpose` of planar images).
- The storage is detected by comparing the `drop` function of the vtable with `drop_shared_vec::<TP::Primitive, CHANNELS>`. rustc merges identical functions by default, so e.g. the instances for `u8` and `i8` can have the same address in optimized builds (not in Miri, where every function gets its own address).
- Example: an `Image<u8, 3>` converted into `Image<i8, 3>` via `DynamicImage` and transposed. The `SharedVecData<u8, 3>` is then read as `SharedVecData<i8, 3>`, and its `Vec<u8>` is taken as a `Vec<i8>`.
- The element values are fine: type changes only happen through `DynamicImage`, which requires `Pod`, and merged functions imply the same size and alignment, so the allocation is freed with the right layout. Non-`Pod` images always have the storage of their own primitive (another one requires `unsafe`).
- What remains is the `Vec` header (pointer, capacity, length): Rust doesn't guarantee that `Vec<A>` and `Vec<B>` have the same layout. In current std it's identical for every `T` (`RawVec<T>` is a type-erased `RawVecInner` and `PhantomData<T>`), only e.g. `-Zrandomize-layout` could break it. `ImageChannelView::into_vec_flat` doesn't have this issue, as it rebuilds the `Vec` with `Vec::from_raw_parts`.
- **Planned direction:** make `SharedVecData` generic over the container of the planar buffer, in particular `Vec<T>` and `Arc<[T]>` (see the TODO in `imbuf/src/arc.rs`):
  - The shared data stores the buffer as raw parts, and taking it out rebuilds the container for the requested primitive (`Vec::from_raw_parts` / `Arc::from_raw`) after checking its size and alignment. No header of a `Vec` of another type is read, which resolves this issue.
  - Planar images could then share an `Arc<[T]>` (`Image::<T, N>::new_arc` for all `N`, today only `N == 1`). `transpose` works in place, if the `Arc` is unique, and keeps the `Arc` storage: `new_arc` planar → `transpose()` → interleaved (Arc channel) → `transpose()` → planar, all on the same shared allocation.

## Nits (not UB)
- `imbuf/src/view.rs` (`slice_mut_channel`): the safety comment says the buffer is "exclusively borrowed for 'a", which isn't true for `view_mut` of a shared channel. The actual invariant is the second sentence: it's only written after `make_mut` of the original.
- `imbuf/src/dynamic.rs` (`make_mut_flat_unchecked`): no `# Safety` section, and the comment "checked after make_mut" describes a `debug_assert!` only.
- `imbuf/src/vec.rs`, `imbuf/src/arc.rs` (`new_vec_flat_unchecked`, `new_arc_flat_unchecked`): public `unsafe fn`s, whose safety sections describe instead of require. External callers can't use a non-`Pod` `T` soundly (only an `ImageChannel` can hide it), so the requirement should say `Pod`, or they could be `pub(crate)`.
- `imbuf-opencv/src/from_imbuf.rs` (`create_unrestricted_mat`): ignores `P::opencv_channels()`, so a `DataType` with several channels gets a `Mat` over a part of the buffer (smaller, not UB).

## Upstream: opencv-rust (not issues of imbuf-opencv)
imbuf-opencv's views are only returned inside `BoxedRef`/`BoxedRefMut`, which limit them to the borrow, like `Mat::new_rows_cols_with_data` of opencv-rust itself, and its imports trust the `Mat` header like opencv-rust's own accessors. opencv-rust breaks both in safe code:
- `OpenCVIntoExternContainer::opencv_into_extern_container` (public, `#[doc(hidden)]`) turns any `BoxedRef` into an owned `Mat`, which outlives the borrow.
- More ways out of a `BoxedRef`/`BoxedRefMut` (from the review): `get_umat(ACCESS_RW)` returns an owned `UMat` over the buffer, `input_array()` + `get_mat()` an owned `Mat`, `core::swap` swaps the header into an owned `Mat`, `assign_mat` assigns it to one, `adjust_roi` returns an owned `Mat`.
- The safe header setters (`set_rows`, `set_cols`, `set_flags`, `set_dims`, `set_u`) let a `Mat` claim more memory than its buffer. `data_typed` then reads out of bounds, and so does an imbuf image created from such a `Mat`. Possible hardening in imbuf-opencv: check `data + len <= datalimit` on import.
- Owned `Mat`s can share a buffer (e.g. `Vector<Mat>::get` returns a shallow copy), so `data_typed_mut(&mut self)` can write under a reader of another `Mat`. The same holds for imbuf images created from `&mut Mat`/`BoxedRefMut` (written without copying). Possible hardening for `&mut Mat`: copy first, if the buffer isn't exclusive (`BoxedRefMut` of ROIs shares the buffer legitimately).

**Next step:** file the upstream issue (draft):

  > **Title:** `BoxedRef`/`BoxedRefMut` can be converted into an owned `Mat` in safe code, which outlives the borrow
  >
  > `BoxedRef<'a, T>` intentionally doesn't implement `Deref`, so the borrowed `Mat` can't be swapped out. But `OpenCVIntoExternContainer::opencv_into_extern_container` (public, `#[doc(hidden)]`, safe) is implemented for `BoxedRef<'_, T>` and returns the inner `T` without the lifetime. For a `Mat` created over borrowed memory (e.g. `Mat::new_rows_cols_with_data(&data)`, which returns `BoxedRef<'_, Mat>`), the result is an owned `Mat`, which points at `data` without a reference count (`u == nullptr`). After `data` is dropped, any access is a use after free, in safe code. `BoxedRefMut` additionally allows writes through the escaped `Mat` after the mutable borrow ended, and `Mat::copy` + the conversion gives two owned `Mat`s writing the same buffer. The same holds for `get_umat(ACCESS_RW)`, `input_array().get_mat()`, `core::swap`, `assign_mat` and `adjust_roi` on a `BoxedRef`/`BoxedRefMut`, and the safe header setters (`set_rows`, `set_cols`, ...) let accessors like `data_typed` read out of bounds.
  >
  > ```rust
  > let mat = {
  >     let data = vec![0u8; 4];
  >     Mat::new_rows_cols_with_data(2, 2, &data).unwrap().opencv_into_extern_container().unwrap()
  > }; // `data` is dropped, `mat` dangles
  > ```
  >
  > Suggestion: don't implement `OpenCVIntoExternContainer` for `BoxedRef`/`BoxedRefMut` (or make it `unsafe`), offer an explicit deep copy instead, and make the header setters `unsafe`.
