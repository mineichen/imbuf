use std::num::NonZeroU32;

use imbuf::{DynamicImage, DynamicImageChannel, Image};
use imbuf_opencv::{ImbufAsOpencv, ImbufAsOpencvMut, OpencvIntoImbuf};
use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, MatTraitManual};

fn dims(width: u32, height: u32) -> (NonZeroU32, NonZeroU32) {
    (
        NonZeroU32::new(width).unwrap(),
        NonZeroU32::new(height).unwrap(),
    )
}

#[test]
fn as_opencv_shares_buffer_for_owned_and_references() {
    let img: Image<u8, 1> = Image::new_vec(
        vec![1, 2, 3, 4],
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(2).unwrap(),
    );
    let ptr = img.buffer().as_ptr() as usize;

    let owned = img.as_opencv().unwrap();
    assert_eq!(owned.data_typed::<u8>().unwrap().as_ptr() as usize, ptr);
    drop(owned);

    let shared: &Image<u8, 1> = &img;
    assert_eq!(
        shared
            .as_opencv()
            .unwrap()
            .data_typed::<u8>()
            .unwrap()
            .as_ptr() as usize,
        ptr
    );

    let mut img = img;
    let exclusive: &mut Image<u8, 1> = &mut img;
    assert_eq!(
        exclusive
            .as_opencv()
            .unwrap()
            .data_typed::<u8>()
            .unwrap()
            .as_ptr() as usize,
        ptr
    );
}

#[test]
fn as_opencv_shares_buffer_for_dynamic_image_references() {
    let dynamic = DynamicImage::from(Image::<u8, 1>::new_vec(
        vec![1, 2, 3, 4],
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(2).unwrap(),
    ));
    let DynamicImageChannel::U8(channel) = dynamic.first() else {
        panic!("Expected U8 channel");
    };
    let ptr = channel.buffer_flat().as_ptr() as usize;

    let shared: &DynamicImage = &dynamic;
    assert_eq!(
        shared
            .as_opencv()
            .unwrap()
            .data_typed::<u8>()
            .unwrap()
            .as_ptr() as usize,
        ptr
    );

    let mut dynamic = dynamic;
    let exclusive: &mut DynamicImage = &mut dynamic;
    assert_eq!(
        exclusive
            .as_opencv()
            .unwrap()
            .data_typed::<u8>()
            .unwrap()
            .as_ptr() as usize,
        ptr
    );
}

#[test]
fn as_opencv_mut_writes_through_for_owned_and_references() {
    let mut img: Image<u8, 1> = Image::new_vec(
        vec![1, 2, 3, 4],
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(2).unwrap(),
    );
    let ptr = img.buffer().as_ptr() as usize;

    {
        let mut view = img.as_opencv_mut().unwrap();
        view.data_typed_mut::<u8>().unwrap()[0] = 9;
    }
    assert_eq!(img.buffer()[0], 9);

    {
        let exclusive: &mut Image<u8, 1> = &mut img;
        let mut view = exclusive.as_opencv_mut().unwrap();
        view.data_typed_mut::<u8>().unwrap()[1] = 8;
    }
    assert_eq!(img.buffer()[1], 8);
    assert_eq!(img.buffer().as_ptr() as usize, ptr);
}

#[test]
fn as_opencv_mut_writes_through_for_dynamic_image() {
    let mut dynamic = DynamicImage::from(Image::<u8, 1>::new_vec(
        vec![1, 2, 3, 4],
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(2).unwrap(),
    ));

    {
        let mut view = dynamic.as_opencv_mut().unwrap();
        view.data_typed_mut::<u8>().unwrap()[0] = 9;
    }

    let imbuf::DynamicImageChannel::U8(channel) = dynamic.first() else {
        panic!("Expected U8 channel");
    };
    assert_eq!(channel.buffer_flat()[0], 9);
}

#[test]
fn opencv_mat_into_dynamic_imbuf_shares_buffer() {
    let mat = Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]]).unwrap();
    let ptr = mat.data() as usize;

    let dynamic = mat.into_imbuf_dynamic().unwrap();
    let DynamicImageChannel::U8(channel) = dynamic.first() else {
        panic!("Expected U8 channel");
    };
    assert_eq!(channel.buffer_flat().as_ptr() as usize, ptr);
    assert_eq!(channel.dimensions(), dims(3, 2));
}

#[test]
fn opencv_mat_into_specific_imbuf() {
    let mat = Mat::from_slice_2d(&[[1u16, 2], [3, 4], [5, 6]]).unwrap();
    let image: Image<u16, 1> = mat.into_imbuf_specific().unwrap();
    assert_eq!(image.dimensions(), dims(2, 3));
    assert_eq!(image.buffer(), &[1, 2, 3, 4, 5, 6]);
}
