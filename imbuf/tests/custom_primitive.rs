use assert_matches::assert_matches;
use std::{
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    panic::{AssertUnwindSafe, catch_unwind},
};

use imbuf::{
    DynamicImage, DynamicImageChannel, DynamicImageChannelView, Image, ImageChannel,
    ImageChannelVTable, PixelTypePrimitive,
};
use testresult::TestResult;

const NZ_1: NonZeroU32 = NonZeroU32::MIN;
const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();
const NZ_U8_1: NonZeroU8 = NonZeroU8::MIN;
const NZ_U8_2: NonZeroU8 = NonZeroU8::new(2).unwrap();
const NZ_USIZE_4: NonZeroUsize = NonZeroUsize::new(4).unwrap();

/// Pod primitive with the same size as `u8`
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
struct Gray(u8);
unsafe impl bytemuck::Zeroable for Gray {}
unsafe impl bytemuck::Pod for Gray {}
impl PixelTypePrimitive for Gray {}

/// Pod primitive with the same size as `f32`, but a smaller alignment
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
struct Rgba8([u8; 4]);
unsafe impl bytemuck::Zeroable for Rgba8 {}
unsafe impl bytemuck::Pod for Rgba8 {}
impl PixelTypePrimitive for Rgba8 {}

/// Primitive which is not Pod, as 0 is not a valid value
#[derive(Clone, Copy, PartialEq, Debug)]
struct NonZero(NonZeroU8);
impl PixelTypePrimitive for NonZero {}

#[test]
fn pod_primitive_round_trips_through_dynamic_image() -> TestResult {
    let image =
        Image::<[Gray; 2], 1>::new_vec(vec![[Gray(1), Gray(2)], [Gray(3), Gray(4)]], NZ_2, NZ_1)?;
    let data_ptr = image.buffer().as_ptr() as usize;
    let expected = image.clone();
    let dynamic = DynamicImage::from(image);
    assert_eq!(dynamic.first().primitive_size().get(), 1);

    let back = Image::<[Gray; 2], 1>::try_from(dynamic)?;
    assert_eq!(back, expected);
    assert_eq!(back.buffer().as_ptr() as usize, data_ptr);
    Ok(())
}

#[test]
fn pod_primitive_can_be_viewed_as_other_primitive_of_same_size() -> TestResult {
    let channel = ImageChannel::<Gray>::new_vec(vec![Gray(1), Gray(2)], NZ_2, NZ_1)?;
    let data_ptr = channel.buffer().as_ptr() as usize;
    let mut dynamic = DynamicImageChannel::from(channel);

    assert_eq!(dynamic.buffer_flat::<u8>(), Ok(&[1, 2][..]));
    dynamic.make_mut_flat::<u8>()?[0] = 3;

    let as_u8 = ImageChannel::<u8>::try_from(dynamic)?;
    assert_eq!(as_u8.buffer(), &[3, 2]);
    assert_eq!(as_u8.buffer().as_ptr() as usize, data_ptr);
    Ok(())
}

#[test]
fn non_pod_primitive_works_with_typed_images() -> TestResult {
    let one = NonZero(NZ_U8_1);
    let two = NonZero(NZ_U8_2);
    let mut image = Image::<NonZero, 2>::new_vec(vec![one, two, two, one], NZ_2, NZ_1)?;
    let clone = image.clone();

    let [first, _] = image.make_mut();
    first[0] = two;

    assert_eq!(image.buffers(), [&[two, two][..], &[two, one][..]]);
    assert_eq!(clone.buffers(), [&[one, two][..], &[two, one][..]]);
    assert_eq!(clone.into_vec(), vec![one, two, two, one]);

    let channel = ImageChannel::<[NonZero; 2]>::new_arc(vec![[one, two]].into(), NZ_1, NZ_1)?;
    assert_eq!(channel.clone().into_vec(), vec![[one, two]]);
    Ok(())
}

#[repr(C, align(4))]
struct AlignedBytes([u8; 8]);
static BYTES: AlignedBytes = AlignedBytes([0, 1, 2, 3, 4, 5, 6, 7]);

unsafe extern "C-unwind" fn panicking_clone<'a>(
    _: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    panic!("The static buffer cannot be cloned")
}
unsafe extern "C-unwind" fn panicking_make_mut(_: &mut DynamicImageChannelView<'_>) {
    panic!("The static buffer is read only")
}
unsafe extern "C-unwind" fn noop_drop(_: &mut DynamicImageChannelView<'_>) {}
unsafe extern "C-unwind" fn four_bytes(_: &DynamicImageChannelView<'_>) -> NonZeroUsize {
    NZ_USIZE_4
}
unsafe extern "C-unwind" fn one_byte(_: &DynamicImageChannelView<'_>) -> NonZeroUsize {
    NonZeroUsize::MIN
}

/// Vtable for `Rgba8` primitives
static STATIC_VTABLE: ImageChannelVTable = ImageChannelVTable {
    clone: panicking_clone,
    make_mut: panicking_make_mut,
    drop: noop_drop,
    primitive_size: four_bytes,
    primitive_alignment: one_byte,
};

/// One `Rgba8` pixel, which starts at an address which is not aligned for `f32`
fn misaligned_channel() -> DynamicImageChannel {
    let ptr = BYTES.0.as_ptr().wrapping_add(1);
    // Safety: The static buffer contains 4 bytes after ptr. Writes are rejected by make_mut
    unsafe {
        DynamicImageChannel::from_raw_parts(
            ptr,
            NZ_1,
            NZ_1,
            NZ_U8_1,
            &STATIC_VTABLE,
            std::ptr::null_mut(),
        )
    }
}

#[test]
fn misaligned_buffer_is_rejected() {
    let channel = misaligned_channel();
    assert_eq!(
        channel.buffer_flat::<Rgba8>(),
        Ok(&[Rgba8([1, 2, 3, 4])][..])
    );
    assert_eq!(channel.buffer_flat_bytes(), &[1, 2, 3, 4]);
    assert_matches!(channel.buffer_flat::<f32>(), Err(_));
    assert_matches!(channel.try_cast::<f32>(), Err(_));
    assert_matches!(ImageChannel::<f32>::try_from(channel), Err(_));
}

#[test]
fn panics_in_vtable_propagate() {
    let mut channel = misaligned_channel();
    assert_matches!(catch_unwind(AssertUnwindSafe(|| channel.clone())), Err(_));
    assert_matches!(
        catch_unwind(AssertUnwindSafe(|| channel
            .make_mut_flat::<Rgba8>()
            .is_ok())),
        Err(_)
    );
}

/// Number of attempts in the alignment tests. Miri places allocations at random addresses, which
/// are only aligned to the requested alignment. Each attempt hits a misaligned copy with a
/// probability of ~1/4 (`Pair` → `Wide`) or ~3/16 (`Bytes4` → `f32`).
const ALIGNMENT_ATTEMPTS: usize = 64;

// The primitives of the alignment tests derive `Pod`, so the tests contain no unsafe code

/// Pod primitive with 4 bytes, but only aligned to 1
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
struct Bytes4([u8; 4]);
impl PixelTypePrimitive for Bytes4 {}

/// Pod primitive with 16 bytes, but only aligned to 8
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
struct Pair([u64; 2]);
impl PixelTypePrimitive for Pair {}

/// Pod primitive with 16 bytes, which is aligned to 16
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, align(16))]
struct Wide([u8; 16]);
impl PixelTypePrimitive for Wide {}

#[test]
fn miri_cloning_typed_view_of_other_primitive_keeps_alignment() -> TestResult {
    // Keep every channel alive, so Miri can't reuse an aligned address for the next allocation
    let mut keep_alive = Vec::new();
    for i in 0..ALIGNMENT_ATTEMPTS {
        let value = u64::try_from(i)?;
        let channel = ImageChannel::<Pair>::new_vec(vec![Pair([value; 2])], NZ_1, NZ_1)?;
        let Ok(wide) = ImageChannel::<Wide>::try_from(DynamicImageChannel::from(channel)) else {
            continue;
        };
        // The Vec vtable clones into an Arc<[Pair]>, which is only aligned for Pair
        let clone = wide.clone();
        assert!(clone.buffer().as_ptr().is_aligned());
        assert_eq!(clone.buffer(), wide.buffer());
        keep_alive.push((wide, clone));
    }
    Ok(())
}

#[test]
fn miri_make_mut_flat_as_other_primitive_keeps_alignment() -> TestResult {
    let mut keep_alive = Vec::new();
    for i in 0..ALIGNMENT_ATTEMPTS {
        let byte = u8::try_from(i)?;
        // CHANNELS > 1 uses the SharedVec storage
        let image = Image::<Bytes4, 2>::new_vec(vec![Bytes4([byte; 4]); 2], NZ_1, NZ_1)?;
        let [first, _] = image.into_channels();
        let mut channel = DynamicImageChannel::from(first);
        // Shares the buffer, so make_mut copies into a new Vec<Bytes4>
        let shared = channel.clone();
        let Ok(buffer) = channel.make_mut_flat::<f32>() else {
            continue;
        };
        assert!(buffer.as_ptr().is_aligned());
        assert_eq!(buffer[0].to_ne_bytes(), [byte; 4]);
        keep_alive.push((channel, shared));
    }
    Ok(())
}
