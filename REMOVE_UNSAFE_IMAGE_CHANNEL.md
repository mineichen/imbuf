Remove UnsafeImageChannel │
│ │
│ Context │
│ │
│ UnsafeImageChannel only exists as a bag of raw fields that is (a) assembled and handed to │
│ from_unsafe, (b) read in vtables via get_unsafe, and (c) mutated in two make_mut │
│ implementations via get_unsafe_mut. Since vtables receive a GenericDynamicImageChannel │
│ that has to be valid again when the function returns, a separate "maybe invalid" type adds API │
│ surface without enabling anything. Both get_unsafe_mut uses (arc make_mut, view │
│ make_original_mut) can instead assign a fully constructed channel (*image = ...), which │
│ drops the old one via its vtable — no channel is ever observable in an invalid state, which also │
│ improves panic safety (a panic between field writes can no longer drop a half-updated channel). │
│ │
│ Design │
│ │
│ - GenericDynamicImageChannel<'a> becomes #[repr(C)] with the fields inlined in the same order │
│ as today (ptr, width, height, vtable, data, pixel_elements) plus a trailing │
│ PhantomData<&'a ()> → FFI layout unchanged. GenericImageChannel stays │
│ repr(transparent) over it. │
│ - Replace UnsafeImageChannel::new_with_vtable + from_unsafe with │
│ pub const unsafe fn from_raw_parts(ptr: *const u8, width, height, pixel_elements, vtable: &'static ImageChannelVTable, data: *mut ()) -> Self │
│ (safety doc = current from_unsafe invariants, incl. validity for 'a). │
│ - Replace get_unsafe with safe associated fns (not methods): │
│ GenericDynamicImageChannel::raw_ptr(ch) -> *const u8, raw_data(ch) -> *mut (), │
│ raw_vtable(ch) -> &'static ImageChannelVTable. │
│ - Remove get_unsafe_mut. Mutation in make_mut = assign a new channel: │
│ - imbuf/src/arc.rs make_mut: *image = DynamicImageChannel::from_arc(new_data, w, h, pe) │
│ (old channel's drop releases its Arc share; drop the ManuallyDrop view without releasing). │
│ - imbuf/src/view.rs make_original_mut: read original from raw_data, call │
│ original.make_exclusive(), then _image = slice_mut_channel::<T>(raw_ptr(original), ..., original) │
│ (old drop is a no-op). Needs T → make it generic make_original_mut::<T>. │
│ - Move calc_len_packed/flat helpers into usage of existing len()/len_flat() │
│ (vec.rs drop_vec_channel uses image.len_flat()). │
│ - Update ImageChannelVTable doc (references to get_unsafe/get_unsafe_mut) and │
│ GenericImageChannel::new_with_vtable docs/impl to call from_raw_parts. │
│ │
│ Files │
│ │
│ - imbuf/src/channel.rs — delete UnsafeImageChannel; update new_with_vtable, view_mut, into_vec_flat, vtable docs. │
│ - imbuf/src/dynamic.rs — struct fields, from_raw_parts, raw accessors, all self.0.x → fields. │
│ - imbuf/src/{vec,arc,shared_vec,view}.rs — constructors + accessors as above. │
│ - imbuf/src/lib.rs — drop export. │
│ - imbuf/tests/custom_primitive.rs, imbuf-opencv/src/into_imbuf.rs — from_raw_parts, raw_data/raw_vtable. │
│ - Docs: imbuf/Releases.md 0.4.0 (replace the UnsafeImageChannel/from_unsafe/get_unsafe_ │
│ bullets with: "Remove UnsafeImageChannel; add unsafe from_raw_parts and raw_ptr/raw_data/raw_vtable; │
│ make_mut replaces the channel by assignment"), RELEASE_CHANGES.md opencv entry. │
│ │
│ Verification │
│ │
│ - cargo fmt --check, cargo clippy --workspace --all-targets --all-features, │
│ cargo test --workspace --all-features (existing arc/view/shared_vec/opencv make_mut tests cover the assignment paths). │
│ - RUSTDOCFLAGS="-D warnings" cargo doc -p imbuf -p imbuf-image --no-deps. │
│ - User: nix run .#miri (unavailable here) for the raw-pointer paths.
