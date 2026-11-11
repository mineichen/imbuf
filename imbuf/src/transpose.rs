//! Transposition between planar buffers (all elements of the first plane, then the second...) and
//! interleaved buffers (all elements of the first pixel, then the second...).
//!
//! The in-place variants split each plane into tiles. They move whole tiles to their target
//! region and transpose the elements within each region with a scratch buffer of a few pages, so
//! all memory accesses are contiguous and the scratch buffer is independent of the image size.
//!
//! The elements of std primitives with 2 to 4 channels are shuffled with SIMD.

use std::{array, mem};

/// Size of the tiles, which the in-place transpositions move as a whole
const TILE_BYTES: usize = 4096;

/// Interleaves the `planes`, which must have the same length
///
/// # Panics
/// If the planes have different lengths
pub(crate) fn interleave<T: Copy + 'static, const C: usize>(planes: [&[T]; C]) -> Vec<[T; C]> {
    let len = planes.first().map_or(0, |plane| plane.len());
    assert!(
        planes.iter().all(|plane| plane.len() == len),
        "The planes have different lengths"
    );
    let Some(&first) = planes.first().and_then(|plane| plane.first()) else {
        return Vec::new();
    };
    // Any value, so the pixels can be written in a single pass over the planes
    let mut pixels = vec![[first; C]; len];
    interleave_into(planes, &mut pixels);
    pixels
}

/// Splits the interleaved `pixels` into `C` planes, which are stored one after another
pub(crate) fn deinterleave<T: Copy + 'static, const C: usize>(pixels: &[[T; C]]) -> Vec<T> {
    let Some(&first) = pixels.first().and_then(|pixel| pixel.first()) else {
        return Vec::new();
    };
    if C == 1 {
        return pixels.as_flattened().to_vec();
    }
    // Any value, so all planes can be written in a single pass over `pixels`
    let mut planar = vec![first; pixels.as_flattened().len()];
    deinterleave_into(pixels, split_planes_mut(&mut planar, pixels.len()));
    planar
}

/// Transposes `C` planes of `buffer.len() / C` elements into interleaved pixels in place
pub(crate) fn interleave_in_place<T: Copy + 'static, const C: usize>(buffer: &mut [T]) {
    interleave_in_place_with_tile::<T, C>(buffer, tile_len::<T>(buffer.len() / C.max(1)));
}

/// Transposes interleaved pixels with `C` elements into `C` planes in place
pub(crate) fn deinterleave_in_place<T: Copy + 'static, const C: usize>(buffer: &mut [T]) {
    deinterleave_in_place_with_tile::<T, C>(buffer, tile_len::<T>(buffer.len() / C.max(1)));
}

/// Writes the elements of the `planes`, which have `pixels.len()` elements each, into `pixels`
fn interleave_into<T: Copy + 'static, const C: usize>(planes: [&[T]; C], pixels: &mut [[T; C]]) {
    let done = simd::interleave(planes, pixels);
    for (i, pixel) in pixels.iter_mut().enumerate().skip(done) {
        *pixel = array::from_fn(|c| planes[c][i]);
    }
}

/// Writes the elements of `pixels` into the `planes`, which have `pixels.len()` elements each
fn deinterleave_into<T: Copy + 'static, const C: usize>(
    pixels: &[[T; C]],
    mut planes: [&mut [T]; C],
) {
    let done = simd::deinterleave(pixels, &mut planes);
    for (i, pixel) in pixels.iter().enumerate().skip(done) {
        for (plane, &value) in planes.iter_mut().zip(pixel) {
            plane[i] = value;
        }
    }
}

/// Splits `planar` into `C` planes of `len` elements
fn split_planes<T, const C: usize>(planar: &[T], len: usize) -> [&[T]; C] {
    array::from_fn(|c| &planar[c * len..][..len])
}

/// Splits `planar` into `C` mutable planes of `len` elements
fn split_planes_mut<T, const C: usize>(planar: &mut [T], len: usize) -> [&mut [T]; C] {
    let mut rest = planar;
    array::from_fn(|_| {
        let (plane, after) = mem::take(&mut rest).split_at_mut(len);
        rest = after;
        plane
    })
}

fn default_tile_len<T>() -> usize {
    (TILE_BYTES / size_of::<T>().max(1)).max(1)
}

/// Number of elements per tile for planes of `plane_len` elements: A divisor of `plane_len`
/// close to the default, if there is one. Then, the planes have no tails, which would have to be
/// rotated. Image sizes usually have such a divisor (e.g. the width).
fn tile_len<T>(plane_len: usize) -> usize {
    let default = default_tile_len::<T>();
    if plane_len <= default {
        return plane_len.max(1);
    }
    (default..=2 * default)
        .chain((default / 2..default).rev())
        .find(|len| plane_len % len == 0)
        .unwrap_or(default)
}

/// The planes of a buffer split into tiles: Each plane consists of a `head` of whole tiles,
/// followed by a `tail`, which is shorter than a tile
struct Tiles {
    /// Number of elements per tile
    len: usize,
    /// Number of tiles per plane
    per_plane: usize,
    /// Number of elements of all tiles of a plane
    head: usize,
    /// Number of elements after the tiles of a plane
    tail: usize,
}

impl Tiles {
    fn new(plane_len: usize, len: usize) -> Self {
        let per_plane = plane_len / len;
        let head = per_plane * len;
        Self {
            len,
            per_plane,
            head,
            tail: plane_len - head,
        }
    }
}

fn interleave_in_place_with_tile<T: Copy + 'static, const C: usize>(
    buffer: &mut [T],
    tile_len: usize,
) {
    if C < 2 {
        return;
    }
    debug_assert_eq!(0, buffer.len() % C, "The planes have different lengths");
    let tiles = Tiles::new(buffer.len() / C, tile_len);

    // [H0 T0 H1 T1 ..] -> [H0 H1 .. T0 T1 ..], where Hc are the heads and Tc the tails
    for c in 1..C {
        // H0..H(c-1) are at the start, followed by T0..T(c-1) and Hc
        buffer[c * tiles.head..][..c * tiles.tail + tiles.head].rotate_left(c * tiles.tail);
    }
    let (heads, tails) = buffer.split_at_mut(C * tiles.head);

    // The tiles of the heads form a matrix with a row per plane. After transposing it, the
    // tiles of pixels at the same position in all planes are adjacent
    transpose_tiles(heads, tiles.len, C, tiles.per_plane);
    let mut scratch = Vec::new();
    for group in heads.chunks_exact_mut(C * tiles.len) {
        interleave_group::<T, C>(group, &mut scratch);
    }
    // The tails contain the pixels after the heads
    interleave_group::<T, C>(tails, &mut scratch);
}

/// Inverse of [`interleave_in_place_with_tile`]
fn deinterleave_in_place_with_tile<T: Copy + 'static, const C: usize>(
    buffer: &mut [T],
    tile_len: usize,
) {
    if C < 2 {
        return;
    }
    debug_assert_eq!(0, buffer.len() % C, "The pixels are incomplete");
    let tiles = Tiles::new(buffer.len() / C, tile_len);

    let (heads, tails) = buffer.split_at_mut(C * tiles.head);
    let mut scratch = Vec::new();
    for group in heads.chunks_exact_mut(C * tiles.len) {
        deinterleave_group::<T, C>(group, &mut scratch);
    }
    deinterleave_group::<T, C>(tails, &mut scratch);
    transpose_tiles(heads, tiles.len, tiles.per_plane, C);

    // [H0 H1 .. T0 T1 ..] -> [H0 T0 H1 T1 ..]
    for c in (1..C).rev() {
        // H0..Hc are at the start, followed by T0..T(c-1)
        buffer[c * tiles.head..][..tiles.head + c * tiles.tail].rotate_right(c * tiles.tail);
    }
}

/// Interleaves `C` planes in `group` in place. The pixels are written into `scratch` first, so
/// the loads from `group` overlap with the shuffles, and then copied back from the cache.
fn interleave_group<T: Copy + 'static, const C: usize>(group: &mut [T], scratch: &mut Vec<T>) {
    resize_scratch(scratch, group);
    let (pixels, rest) = scratch.as_chunks_mut::<C>();
    debug_assert!(rest.is_empty(), "The planes have different lengths");
    interleave_into(split_planes(group, pixels.len()), pixels);
    group.copy_from_slice(scratch);
}

/// Splits the pixels with `C` elements in `group` into planes in place, using `scratch` like
/// [`interleave_group`]
fn deinterleave_group<T: Copy + 'static, const C: usize>(group: &mut [T], scratch: &mut Vec<T>) {
    resize_scratch(scratch, group);
    let (pixels, rest) = group.as_chunks::<C>();
    debug_assert!(rest.is_empty(), "The pixels are incomplete");
    deinterleave_into(pixels, split_planes_mut(scratch, pixels.len()));
    group.copy_from_slice(scratch);
}

/// Resizes `scratch` to the length of `group`. Its values are overwritten, so they are only
/// initialized, if the length changes (once for all groups of the same length).
fn resize_scratch<T: Copy>(scratch: &mut Vec<T>, group: &[T]) {
    if scratch.len() != group.len() {
        scratch.clear();
        scratch.extend_from_slice(group);
    }
}

/// Transposes the `rows` x `cols` matrix of tiles with `tile_len` elements each (row major) in
/// place. Each tile is moved once along the cycles of the permutation.
fn transpose_tiles<T: Copy>(buffer: &mut [T], tile_len: usize, rows: usize, cols: usize) {
    if rows < 2 || cols < 2 {
        // A vector is its own transpose
        return;
    }
    let count = rows * cols;
    debug_assert_eq!(count * tile_len, buffer.len());
    // The transposed matrix has `rows` columns. Its tile `j` is the tile `source(j)` of the
    // original matrix, which has `cols` columns.
    let source = |j: usize| (j % rows) * cols + j / rows;
    let mut moved = vec![false; count];
    let mut saved = Vec::with_capacity(tile_len);
    for start in 0..count {
        if moved[start] || source(start) == start {
            continue;
        }
        saved.clear();
        saved.extend_from_slice(&buffer[start * tile_len..][..tile_len]);
        let mut target = start;
        loop {
            moved[target] = true;
            let from = source(target);
            if from == start {
                break;
            }
            buffer.copy_within(from * tile_len..(from + 1) * tile_len, target * tile_len);
            target = from;
        }
        buffer[target * tile_len..][..tile_len].copy_from_slice(&saved);
    }
}

/// SIMD versions of [`interleave_into`] and [`deinterleave_into`]. They shuffle the bytes of
/// blocks of 16 bytes per plane and return the number of pixels, which they transposed. The
/// remaining pixels are left to the caller.
mod simd {
    #![allow(
        clippy::needless_range_loop,
        reason = "Plain loops are unrolled even if generic helpers of std aren't inlined"
    )]

    use std::{any::TypeId, slice};

    use fearless_simd::{Level, Simd, dispatch, prelude::*, u8x16, u8x32};

    /// Number of bytes per vector
    const LANES: usize = 16;

    pub(super) fn interleave<T: 'static, const C: usize>(
        planes: [&[T]; C],
        pixels: &mut [[T; C]],
    ) -> usize {
        if !supports::<T, C>() {
            return 0;
        }
        // Safety: `T` is a primitive of std (checked above)
        let planes = planes.map(|plane| unsafe { bytes(plane) });
        // Safety: `T` is a primitive of std (checked above)
        let pixels = unsafe { bytes_mut(pixels.as_flattened_mut()) };
        let level = Level::new();
        match size_of::<T>() {
            1 => dispatch!(level, simd => interleave_bytes::<_, C, 1>(simd, planes, pixels)),
            2 => dispatch!(level, simd => interleave_bytes::<_, C, 2>(simd, planes, pixels)),
            4 => dispatch!(level, simd => interleave_bytes::<_, C, 4>(simd, planes, pixels)),
            8 => dispatch!(level, simd => interleave_bytes::<_, C, 8>(simd, planes, pixels)),
            _ => 0,
        }
    }

    pub(super) fn deinterleave<T: 'static, const C: usize>(
        pixels: &[[T; C]],
        planes: &mut [&mut [T]; C],
    ) -> usize {
        if !supports::<T, C>() {
            return 0;
        }
        // Safety: `T` is a primitive of std (checked above)
        let pixels = unsafe { bytes(pixels.as_flattened()) };
        // Safety: `T` is a primitive of std (checked above)
        let mut planes = planes.each_mut().map(|plane| unsafe { bytes_mut(plane) });
        let level = Level::new();
        let planes = &mut planes;
        match size_of::<T>() {
            1 => dispatch!(level, simd => deinterleave_bytes::<_, C, 1>(simd, pixels, planes)),
            2 => dispatch!(level, simd => deinterleave_bytes::<_, C, 2>(simd, pixels, planes)),
            4 => dispatch!(level, simd => deinterleave_bytes::<_, C, 4>(simd, pixels, planes)),
            8 => dispatch!(level, simd => deinterleave_bytes::<_, C, 8>(simd, pixels, planes)),
            _ => 0,
        }
    }

    // The kernels only use plain loops and indexing: Generic helpers of std (e.g. `array::map`)
    // aren't always inlined (e.g. with several codegen units), which would turn the constant
    // tables into variables and prevent unrolling.

    /// Interleaves pairs of blocks of `LANES` bytes of all planes, which have the same length,
    /// and returns the number of pixels with `C` elements of `E` bytes
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "Only inlined code gets the target features of the SIMD level"
    )]
    fn interleave_bytes<S: Simd, const C: usize, const E: usize>(
        simd: S,
        planes: [&[u8]; C],
        pixels: &mut [u8],
    ) -> usize {
        let tables = const { &WideTables::new(&interleave_tables::<C, E>()) };
        let pairs = planes[0].len() / (2 * LANES);
        let mut inputs = [u8x32::splat(simd, 0); C];
        for pair in 0..pairs {
            for c in 0..C {
                inputs[c] = u8x32::from_slice(simd, &planes[c][pair * 2 * LANES..][..2 * LANES]);
            }
            // The first half of each vector belongs to the first block of pixels
            let output = &mut pixels[pair * 2 * C * LANES..][..2 * C * LANES];
            for j in 0..C {
                let (first, second) = tables.shuffle(simd, &inputs, j).split();
                first.store_slice(&mut output[j * LANES..][..LANES]);
                second.store_slice(&mut output[(C + j) * LANES..][..LANES]);
            }
        }
        pairs * 2 * (LANES / E)
    }

    /// Splits pairs of blocks of `C * LANES` bytes into the planes and returns the number of
    /// pixels with `C` elements of `E` bytes
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "Only inlined code gets the target features of the SIMD level"
    )]
    fn deinterleave_bytes<S: Simd, const C: usize, const E: usize>(
        simd: S,
        pixels: &[u8],
        planes: &mut [&mut [u8]; C],
    ) -> usize {
        let tables = const { &WideTables::new(&deinterleave_tables::<C, E>()) };
        let pairs = pixels.len() / (2 * C * LANES);
        let mut inputs = [u8x32::splat(simd, 0); C];
        for pair in 0..pairs {
            let input = &pixels[pair * 2 * C * LANES..][..2 * C * LANES];
            for j in 0..C {
                let first = u8x16::from_slice(simd, &input[j * LANES..][..LANES]);
                let second = u8x16::from_slice(simd, &input[(C + j) * LANES..][..LANES]);
                inputs[j] = first.combine(second);
            }
            for c in 0..C {
                let plane = &mut planes[c][pair * 2 * LANES..][..2 * LANES];
                tables.shuffle(simd, &inputs, c).store_slice(plane);
            }
        }
        pairs * 2 * (LANES / E)
    }

    /// Tables of a kernel for vectors of two blocks of `LANES` bytes, which are shuffled
    /// independently
    struct WideTables<const C: usize> {
        /// `indices[j][c]` selects the bytes of output vector `j` from input vector `c`
        indices: [[[u8; 2 * LANES]; C]; C],
        /// Keeps the selected bytes, as out of range indices don't select zero on all SIMD
        /// levels
        masks: [[[u8; 2 * LANES]; C]; C],
    }

    impl<const C: usize> WideTables<C> {
        /// Uses the `tables` of a single block (`u8::MAX` for unselected bytes) for both blocks
        const fn new(tables: &[[[u8; LANES]; C]; C]) -> Self {
            let mut indices = [[[0; 2 * LANES]; C]; C];
            let mut masks = [[[0; 2 * LANES]; C]; C];
            let mut j = 0;
            while j < C {
                let mut c = 0;
                while c < C {
                    let mut byte = 0;
                    while byte < 2 * LANES {
                        let index = tables[j][c][byte % LANES];
                        if (index as usize) < LANES {
                            indices[j][c][byte] = index;
                            masks[j][c][byte] = u8::MAX;
                        }
                        byte += 1;
                    }
                    c += 1;
                }
                j += 1;
            }
            Self { indices, masks }
        }

        /// Combines the bytes, which the tables of output vector `j` select from the `inputs`
        #[inline(always)]
        #[allow(
            clippy::inline_always,
            reason = "Only inlined code gets the target features of the SIMD level"
        )]
        fn shuffle<S: Simd>(&self, simd: S, inputs: &[u8x32<S>; C], j: usize) -> u8x32<S> {
            let mut output = u8x32::splat(simd, 0);
            for c in 0..C {
                let indices = u8x32::from_slice(simd, &self.indices[j][c]);
                let mask = u8x32::from_slice(simd, &self.masks[j][c]);
                output |= inputs[c].swizzle_dyn_within_blocks(indices) & mask;
            }
            output
        }
    }

    /// `tables[j][c]` selects the bytes of output vector `j` from plane `c`, if `C` blocks of
    /// `LANES` bytes with `E` bytes per element are interleaved
    #[allow(clippy::cast_possible_truncation, reason = "Indices are below LANES")]
    const fn interleave_tables<const C: usize, const E: usize>() -> [[[u8; LANES]; C]; C] {
        let mut tables = [[[u8::MAX; LANES]; C]; C];
        let mut output = 0;
        while output < C * LANES {
            let pixel = output / (C * E);
            let channel = output % (C * E) / E;
            let input = pixel * E + output % E;
            tables[output / LANES][channel][output % LANES] = input as u8;
            output += 1;
        }
        tables
    }

    /// `tables[c][j]` selects the bytes of plane `c` from input vector `j`, if a block of
    /// `C * LANES` interleaved bytes with `E` bytes per element is split into planes
    #[allow(clippy::cast_possible_truncation, reason = "Indices are below LANES")]
    const fn deinterleave_tables<const C: usize, const E: usize>() -> [[[u8; LANES]; C]; C] {
        let mut tables = [[[u8::MAX; LANES]; C]; C];
        let mut channel = 0;
        while channel < C {
            let mut output = 0;
            while output < LANES {
                let input = output / E * C * E + channel * E + output % E;
                tables[channel][input / LANES][output] = (input % LANES) as u8;
                output += 1;
            }
            channel += 1;
        }
        tables
    }

    /// Whether the elements are transposed with SIMD: Only primitives of std can be viewed as
    /// bytes, as they have no padding and any bit pattern is valid. More channels would require
    /// more shuffles per byte.
    fn supports<T: 'static, const C: usize>() -> bool {
        let primitives = [
            TypeId::of::<u8>(),
            TypeId::of::<i8>(),
            TypeId::of::<u16>(),
            TypeId::of::<i16>(),
            TypeId::of::<u32>(),
            TypeId::of::<i32>(),
            TypeId::of::<u64>(),
            TypeId::of::<i64>(),
            TypeId::of::<usize>(),
            TypeId::of::<isize>(),
            TypeId::of::<f32>(),
            TypeId::of::<f64>(),
        ];
        (2..=4).contains(&C) && primitives.contains(&TypeId::of::<T>())
    }

    /// # Safety
    /// `T` has to be a primitive of std (see [`supports`])
    unsafe fn bytes<T>(slice: &[T]) -> &[u8] {
        // Safety: Primitives of std have no padding (precondition)
        unsafe { slice::from_raw_parts(slice.as_ptr().cast(), size_of_val(slice)) }
    }

    /// # Safety
    /// `T` has to be a primitive of std (see [`supports`])
    unsafe fn bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
        // Safety: Primitives of std have no padding and any bit pattern is a valid value
        // (precondition)
        unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), size_of_val(slice)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_interleave<const C: usize>(planar: &[u16]) -> Vec<u16> {
        let len = planar.len() / C;
        (0..len)
            .flat_map(|i| (0..C).map(move |c| planar[c * len + i]))
            .collect()
    }

    /// Checks all combinations of plane and tile lengths up to the given maxima
    fn check_in_place_round_trip<const C: usize>(max_plane_len: usize, max_tile_len: usize) {
        for plane_len in 0..=max_plane_len {
            let planar = (0..u16::try_from(C * plane_len).unwrap()).collect::<Vec<_>>();
            let expected = naive_interleave::<C>(&planar);
            for tile_len in 1..=max_tile_len {
                let mut buffer = planar.clone();
                interleave_in_place_with_tile::<u16, C>(&mut buffer, tile_len);
                assert_eq!(
                    buffer, expected,
                    "interleave {C}x{plane_len}, tile {tile_len}"
                );

                deinterleave_in_place_with_tile::<u16, C>(&mut buffer, tile_len);
                assert_eq!(
                    buffer, planar,
                    "deinterleave {C}x{plane_len}, tile {tile_len}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "Slow and only safe code")]
    fn in_place_with_all_small_tile_lengths() {
        check_in_place_round_trip::<1>(5, 3);
        check_in_place_round_trip::<2>(17, 5);
        check_in_place_round_trip::<3>(17, 5);
        check_in_place_round_trip::<4>(13, 5);
    }

    #[test]
    #[cfg_attr(miri, ignore = "Slow and only safe code")]
    fn in_place_with_default_tile_len() {
        // Two whole tiles and a tail per plane
        let plane_len = 2 * default_tile_len::<u64>() + 3;
        let planar = (0..3 * plane_len as u64).collect::<Vec<_>>();
        let mut buffer = planar.clone();

        interleave_in_place::<u64, 3>(&mut buffer);
        let pixels = buffer.as_chunks::<3>().0;
        assert_eq!(pixels.len(), plane_len);
        for (i, pixel) in pixels.iter().enumerate() {
            let i = i as u64;
            let plane_len = plane_len as u64;
            assert_eq!(*pixel, [i, plane_len + i, 2 * plane_len + i]);
        }

        deinterleave_in_place::<u64, 3>(&mut buffer);
        assert_eq!(buffer, planar);
    }

    /// Compares the (SIMD) transposition of planes with up to `max_len` elements to a naive one
    fn check_transposition<T, const C: usize>(max_len: usize, element: impl Fn(usize) -> T)
    where
        T: Copy + PartialEq + std::fmt::Debug + 'static,
    {
        // Miri only has to check the unsafe byte views, which don't depend on the length
        let min_len = if cfg!(miri) { max_len } else { 0 };
        for len in min_len..=max_len {
            let planar = (0..C * len).map(&element).collect::<Vec<_>>();
            let planes = split_planes::<T, C>(&planar, len);
            let expected = (0..len)
                .map(|i| array::from_fn(|c| planes[c][i]))
                .collect::<Vec<[T; C]>>();

            assert_eq!(interleave(planes), expected, "interleave {C}x{len}");
            assert_eq!(deinterleave(&expected), planar, "deinterleave {C}x{len}");
        }
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss
    )]
    fn transposition_of_all_element_sizes() {
        // Several blocks of 16 bytes per plane and a remainder
        check_transposition::<u8, 2>(40, |i| i as u8);
        check_transposition::<u8, 3>(40, |i| i as u8);
        check_transposition::<i8, 4>(40, |i| i as i8);
        check_transposition::<u16, 3>(20, |i| i as u16);
        check_transposition::<i16, 4>(20, |i| i as i16);
        check_transposition::<f32, 3>(10, |i| i as f32);
        check_transposition::<u32, 4>(10, |i| i as u32);
        check_transposition::<u64, 2>(6, |i| i as u64);
        check_transposition::<f64, 3>(6, |i| i as f64);
        check_transposition::<i64, 4>(6, |i| i as i64);
        // Without SIMD
        check_transposition::<u8, 5>(40, |i| i as u8);
        check_transposition::<[u8; 2], 3>(20, |i| [i as u8, 0]);
    }

    #[test]
    fn simd_transposes_whole_blocks() {
        // 2 blocks of 16 bytes per plane and a remainder of 3 elements
        let planar = (0..3 * 35).collect::<Vec<u8>>();
        let mut pixels = vec![[0; 3]; 35];
        assert_eq!(simd::interleave(split_planes(&planar, 35), &mut pixels), 32);

        let mut planar = vec![0u8; 3 * 35];
        let planes = &mut split_planes_mut(&mut planar, 35);
        assert_eq!(simd::deinterleave(&pixels, planes), 32);
        // Custom primitives aren't viewed as bytes
        assert_eq!(
            simd::interleave([&[[0u8; 2]; 35][..]; 3], &mut [[[0; 2]; 3]; 35]),
            0
        );
    }

    #[test]
    fn copying_transposition() {
        let pixels = interleave([&[1u8, 2][..], &[3, 4], &[5, 6]]);
        assert_eq!(pixels, [[1, 3, 5], [2, 4, 6]]);
        assert_eq!(deinterleave(&pixels), [1, 2, 3, 4, 5, 6]);
        assert_eq!(deinterleave::<u8, 1>(&[[1], [2]]), [1, 2]);
        assert_eq!(deinterleave::<u8, 3>(&[]), []);
    }

    #[test]
    fn tile_len_avoids_tails() {
        assert_eq!(tile_len::<u8>(4096 * 3000), 4096);
        // 4095 x 3001 pixels with a divisor below the default
        assert_eq!(tile_len::<u8>(4095 * 3001), 4095);
        // 1920 x 1080 pixels with a divisor above the default of 2048 `u16`s
        assert_eq!(tile_len::<u16>(1920 * 1080), 2160);
        assert_eq!(tile_len::<u8>(17), 17);
        // A prime has no divisor near the default
        assert_eq!(tile_len::<u8>(1_000_003), 4096);
    }

    #[test]
    #[should_panic(expected = "different lengths")]
    fn interleaving_planes_of_different_lengths_panics() {
        let _ = interleave([&[1u8, 2][..], &[3]]);
    }
}
