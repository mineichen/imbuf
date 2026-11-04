//! `buffer_mut` and `buffer_flat_mut` only return buffers, which can be written without copying

use assert_matches::assert_matches;
use std::num::{NonZeroU8, NonZeroU32};

use imbuf::{
    DynamicImage, DynamicImageChannel, Image, ImageChannel, ImageChannelView,
    IncompatibleImageError,
};
use testresult::TestResult;

const NZ_1: NonZeroU32 = NonZeroU32::MIN;
const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

#[test]
fn vec_channel_is_written_without_copy() -> TestResult {
    let mut channel = ImageChannel::<[u8; 2]>::new_vec(vec![[1, 2], [3, 4]], NZ_2, NZ_1)?;
    let ptr = channel.buffer().as_ptr();

    let buffer = channel.buffer_mut().ok_or("Vec is exclusive")?;
    assert_eq!(buffer.as_ptr(), ptr);
    buffer[0] = [5, 6];
    channel.buffer_flat_mut().ok_or("Vec is exclusive")?[3] = 7;

    assert_eq!(channel.buffer(), &[[5, 6], [3, 7]]);
    assert_eq!(channel.buffer().as_ptr(), ptr);
    Ok(())
}

#[test]
fn shared_arc_is_mutable_once_unique() -> TestResult {
    let mut channel = ImageChannel::<u8>::new_arc(vec![1, 2].into(), NZ_2, NZ_1)?;
    let ptr = channel.buffer().as_ptr();
    assert!(channel.buffer_mut().is_some());

    let mut clone = channel.clone();
    assert_eq!(channel.buffer_mut(), None);
    assert_eq!(clone.buffer_flat_mut(), None);
    // Nothing was copied
    assert_eq!(channel.buffer().as_ptr(), ptr);
    assert_eq!(clone.buffer().as_ptr(), ptr);

    drop(clone);
    let buffer = channel.buffer_mut().ok_or("The Arc is unique")?;
    assert_eq!(buffer.as_ptr(), ptr);
    Ok(())
}

#[test]
fn clone_of_vec_channel_is_mutable() -> TestResult {
    let mut channel = ImageChannel::<u8>::new_vec(vec![1, 2], NZ_2, NZ_1)?;
    // Copies the Vec into a unique Arc
    let mut clone = channel.clone();
    clone.buffer_mut().ok_or("The Arc is unique")?[0] = 3;
    channel.buffer_mut().ok_or("The Vec is exclusive")?[0] = 4;
    assert_eq!(clone.buffer(), &[3, 2]);
    assert_eq!(channel.buffer(), &[4, 2]);
    Ok(())
}

#[test]
fn shared_slice_is_never_mutable() -> TestResult {
    let data = [1u8, 2];
    let mut view = ImageChannelView::new_slice(&data, NZ_2, NZ_1)?;
    assert_eq!(view.buffer_mut(), None);
    assert_eq!(view.buffer_flat_mut(), None);
    assert_eq!(view.buffer().as_ptr(), data.as_ptr());
    Ok(())
}

#[test]
fn exclusive_slice_is_written_through() -> TestResult {
    let mut data = [1u8, 2];
    let mut view = ImageChannelView::new_slice_mut(&mut data, NZ_2, NZ_1)?;
    view.buffer_mut()
        .ok_or("The slice is borrowed exclusively")?[0] = 3;
    drop(view);
    assert_eq!(data, [3, 2]);
    Ok(())
}

#[test]
fn view_mut_writes_into_exclusive_original() -> TestResult {
    let mut channel = ImageChannel::<u8>::new_vec(vec![1, 2], NZ_2, NZ_1)?;
    let ptr = channel.buffer().as_ptr();
    let mut view = channel.view_mut();
    let buffer = view.buffer_mut().ok_or("The original is exclusive")?;
    assert_eq!(buffer.as_ptr(), ptr);
    buffer[1] = 3;
    drop(view);
    assert_eq!(channel.buffer(), &[1, 3]);
    Ok(())
}

#[test]
fn view_mut_of_shared_original_is_not_mutable() -> TestResult {
    let mut channel = ImageChannel::<u8>::new_arc(vec![1, 2].into(), NZ_2, NZ_1)?;
    let clone = channel.clone();
    let mut view = channel.view_mut();
    assert_eq!(view.buffer_mut(), None);
    drop(view);
    // The original still shares the buffer with the clone
    assert_eq!(channel.buffer().as_ptr(), clone.buffer().as_ptr());

    drop(clone);
    let mut view = channel.view_mut();
    view.buffer_mut().ok_or("The original is unique")?[0] = 3;
    drop(view);
    assert_eq!(channel.buffer(), &[3, 2]);
    Ok(())
}

#[test]
fn planar_image_is_mutable_until_a_channel_is_shared() -> TestResult {
    // Multiple channels share the planar Vec
    let mut image = Image::<u8, 3>::new_vec(vec![1, 2, 3, 4, 5, 6], NZ_2, NZ_1)?;
    let ptrs = image.buffers().map(<[u8]>::as_ptr);
    let [r, g, b] = image.buffers_mut().ok_or("The planes are exclusive")?;
    assert_eq!([r.as_ptr(), g.as_ptr(), b.as_ptr()], ptrs);
    r[0] = 7;
    b[1] = 8;
    assert_eq!(image.buffers(), [&[7, 2][..], &[3, 4], &[5, 8]]);

    let green = image.clone().into_channels()[1].clone();
    assert_eq!(image.buffers_mut(), None);
    assert_eq!(image.buffers().map(<[u8]>::as_ptr), ptrs);

    drop(green);
    assert!(image.buffers_mut().is_some());
    Ok(())
}

#[test]
fn single_channel_image() -> TestResult {
    let mut image = Image::<[u8; 2], 1>::new_vec(vec![[1, 2]], NZ_1, NZ_1)?;
    image.buffer_mut().ok_or("The Vec is exclusive")?[0] = [3, 4];
    image.buffer_flat_mut().ok_or("The Vec is exclusive")?[1] = 5;
    assert_eq!(image.buffer(), &[[3, 5]]);

    let mut view = image.view();
    assert_eq!(view.buffer_mut(), None);
    assert_eq!(view.buffer_flat_mut(), None);
    Ok(())
}

#[test]
fn dynamic_channel_checks_the_layout() -> TestResult {
    let mut channel = DynamicImageChannel::new_vec(vec![[1u8, 2], [3, 4]], NZ_2, NZ_1)?;
    channel
        .buffer_mut::<[u8; 2]>()?
        .ok_or("The Vec is exclusive")?[0] = [5, 6];
    channel
        .buffer_flat_mut::<i8>()?
        .ok_or("The Vec is exclusive")?[3] = -1;
    assert_eq!(channel.buffer::<[u8; 2]>()?, &[[5, 6], [3, 255]]);

    // `pixel_elements` doesn't match
    assert_matches!(channel.buffer_mut::<u8>(), Err(_));
    // Size doesn't match
    assert_matches!(channel.buffer_mut::<[u16; 2]>(), Err(_));
    assert_matches!(channel.buffer_flat_mut::<u16>(), Err(_));

    // The clone of a Vec channel copies it into an Arc, whose clones share it
    let mut arc = channel.clone();
    let shared = arc.clone();
    assert_matches!(arc.buffer_mut::<[u8; 2]>(), Ok(None));
    assert_matches!(arc.buffer_flat_mut::<u8>(), Ok(None));
    drop(shared);
    assert_matches!(arc.buffer_flat_mut::<u8>(), Ok(Some(_)));
    Ok(())
}

#[test]
fn dynamic_image_requires_a_single_channel() -> TestResult {
    let mut image = DynamicImage::from(Image::<u16, 1>::new_vec(vec![1, 2], NZ_2, NZ_1)?);
    image.buffer_mut::<u16>()?.ok_or("The Vec is exclusive")?[0] = 3;
    // Viewed as another primitive with the same layout
    image
        .buffer_flat_mut::<i16>()?
        .ok_or("The Vec is exclusive")?[1] = -1;
    assert_eq!(image[0].buffer::<u16>()?, &[3, u16::MAX]);

    assert_matches!(
        image.buffer_mut::<u8>(),
        Err(IncompatibleImageError::Channel(_))
    );
    // Shares the Arc, into which the clone of the image copied the Vec
    let mut arc = image.clone();
    let shared = arc.clone();
    assert_matches!(arc.buffer_mut::<u16>(), Ok(None));
    assert_matches!(arc.buffer_flat_mut::<u16>(), Ok(None));
    drop(shared);
    assert_matches!(arc.buffer_mut::<u16>(), Ok(Some(_)));

    let mut planar = DynamicImage::from(Image::<u16, 2>::new_vec(vec![1, 2], NZ_1, NZ_1)?);
    assert_matches!(
        planar.buffer_mut::<u16>(),
        Err(IncompatibleImageError::ChannelCount(_))
    );
    assert_matches!(
        planar.buffer_flat_mut::<u16>(),
        Err(IncompatibleImageError::ChannelCount(_))
    );
    let [first, second] = planar
        .try_cast_mut::<u16, 2>()?
        .buffers_mut()
        .ok_or("The planes are exclusive")?;
    first[0] = 3;
    second[0] = 4;
    assert_eq!(planar[1].buffer::<u16>()?, &[4]);
    Ok(())
}

#[test]
fn pixel_elements_are_kept() -> TestResult {
    let mut channel = DynamicImageChannel::new_vec_flat(
        vec![1u8, 2, 3, 4],
        NZ_2,
        NZ_1,
        NonZeroU8::new(2).unwrap(),
    )?;
    assert_eq!(
        channel.buffer_mut::<[u8; 2]>()?.map(|pixels| pixels.len()),
        Some(2)
    );
    assert_eq!(
        channel
            .buffer_flat_mut::<u8>()?
            .map(|primitives| primitives.len()),
        Some(4)
    );
    Ok(())
}
