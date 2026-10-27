use std::{
    num::{NonZeroU8, NonZeroU32, NonZeroUsize},
    panic::{AssertUnwindSafe, catch_unwind},
};

use imbuf::{
    DynamicImage, DynamicImageChannel, DynamicImageChannelView, Image, ImageChannel,
    ImageChannelVTable, PixelTypePrimitive,
};

const ONE: NonZeroU32 = NonZeroU32::MIN;
const TWO: NonZeroU32 = NonZeroU32::new(2).unwrap();

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
fn pod_primitive_round_trips_through_dynamic_image() {
    let image =
        Image::<[Gray; 2], 1>::new_vec(vec![[Gray(1), Gray(2)], [Gray(3), Gray(4)]], TWO, ONE);
    let data_ptr = image.buffer().as_ptr() as usize;
    let expected = image.clone();
    let dynamic = DynamicImage::from(image);
    assert_eq!(dynamic.first().primitive_size().get(), 1);

    let back = Image::<[Gray; 2], 1>::try_from(dynamic).unwrap();
    assert_eq!(back, expected);
    assert_eq!(back.buffer().as_ptr() as usize, data_ptr);
}

#[test]
fn pod_primitive_can_be_viewed_as_other_primitive_of_same_size() {
    let channel = ImageChannel::<Gray>::new_vec(vec![Gray(1), Gray(2)], TWO, ONE);
    let data_ptr = channel.buffer().as_ptr() as usize;
    let mut dynamic = DynamicImageChannel::from(channel);

    assert_eq!(dynamic.buffer_flat::<u8>(), Some(&[1, 2][..]));
    dynamic.make_mut_flat::<u8>().unwrap()[0] = 3;

    let as_u8 = ImageChannel::<u8>::try_from(dynamic).unwrap();
    assert_eq!(as_u8.buffer(), &[3, 2]);
    assert_eq!(as_u8.buffer().as_ptr() as usize, data_ptr);
}

#[test]
fn non_pod_primitive_works_with_typed_images() {
    let one = NonZero(NonZeroU8::MIN);
    let two = NonZero(NonZeroU8::new(2).unwrap());
    let mut image = Image::<NonZero, 2>::new_vec(vec![one, two, two, one], TWO, ONE);
    let clone = image.clone();

    let [first, _] = image.make_mut();
    first[0] = two;

    assert_eq!(image.buffers(), [&[two, two][..], &[two, one][..]]);
    assert_eq!(clone.buffers(), [&[one, two][..], &[two, one][..]]);
    assert_eq!(clone.into_vec(), vec![one, two, two, one]);

    let channel = ImageChannel::<[NonZero; 2]>::new_arc(vec![[one, two]].into(), ONE, ONE);
    assert_eq!(channel.clone().into_vec(), vec![[one, two]]);
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
extern "C-unwind" fn four_bytes() -> NonZeroUsize {
    NonZeroUsize::new(4).unwrap()
}

static STATIC_VTABLE: ImageChannelVTable = ImageChannelVTable {
    clone: panicking_clone,
    make_mut: panicking_make_mut,
    drop: noop_drop,
    primitive_size: four_bytes,
};

/// One `Rgba8` pixel, which starts at an address which is not aligned for `f32`
fn misaligned_channel() -> DynamicImageChannel {
    let ptr = BYTES.0.as_ptr().wrapping_add(1);
    // Safety: The static buffer contains 4 bytes after ptr. Writes are rejected by make_mut
    unsafe {
        DynamicImageChannel::from_raw_parts(
            ptr,
            ONE,
            ONE,
            NonZeroU8::MIN,
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
        Some(&[Rgba8([1, 2, 3, 4])][..])
    );
    assert_eq!(channel.buffer_flat_bytes(), &[1, 2, 3, 4]);
    assert_eq!(channel.buffer_flat::<f32>(), None);
    assert!(channel.try_cast::<f32>().is_none());
    assert!(ImageChannel::<f32>::try_from(channel).is_err());
}

#[test]
fn panics_in_vtable_propagate() {
    let mut channel = misaligned_channel();
    assert!(catch_unwind(AssertUnwindSafe(|| channel.clone())).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| channel
            .make_mut_flat::<Rgba8>()
            .is_some()))
        .is_err()
    );
}
