use std::num::NonZeroU32;

const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

use testresult::TestResult;

use imbuf::{DynamicImage, Image};
use imbuf_opencv::{ImbufAsOpencv, ImbufAsOpencvMut, OpencvIntoImbuf};
use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, MatTraitManual};

fn dims(width: u32, height: u32) -> TestResult<(NonZeroU32, NonZeroU32)> {
    Ok((
        NonZeroU32::new(width).ok_or("zero width")?,
        NonZeroU32::new(height).ok_or("zero height")?,
    ))
}

#[test]
fn as_opencv_shares_buffer_for_owned_and_references() -> TestResult {
    let img: Image<u8, 1> = Image::new_vec(vec![1, 2, 3, 4], NZ_2, NZ_2)?;
    let ptr = img.buffer().as_ptr() as usize;

    let owned = img.as_opencv()?;
    assert_eq!(owned.data_typed::<u8>()?.as_ptr() as usize, ptr);
    drop(owned);

    let shared: &Image<u8, 1> = &img;
    assert_eq!(
        shared.as_opencv()?.data_typed::<u8>()?.as_ptr() as usize,
        ptr
    );

    let mut img = img;
    let exclusive: &mut Image<u8, 1> = &mut img;
    assert_eq!(
        exclusive.as_opencv()?.data_typed::<u8>()?.as_ptr() as usize,
        ptr
    );
    Ok(())
}

#[test]
fn as_opencv_shares_buffer_for_dynamic_image_references() -> TestResult {
    let dynamic = DynamicImage::from(Image::<u8, 1>::new_vec(vec![1, 2, 3, 4], NZ_2, NZ_2)?);
    let channel = dynamic.first().try_cast::<u8>()?;
    let ptr = channel.buffer_flat().as_ptr() as usize;

    let shared: &DynamicImage = &dynamic;
    assert_eq!(
        shared.as_opencv()?.data_typed::<u8>()?.as_ptr() as usize,
        ptr
    );

    let mut dynamic = dynamic;
    let exclusive: &mut DynamicImage = &mut dynamic;
    assert_eq!(
        exclusive.as_opencv()?.data_typed::<u8>()?.as_ptr() as usize,
        ptr
    );
    Ok(())
}

#[test]
fn as_opencv_mut_writes_through_for_owned_and_references() -> TestResult {
    let mut img: Image<u8, 1> = Image::new_vec(vec![1, 2, 3, 4], NZ_2, NZ_2)?;
    let ptr = img.buffer().as_ptr() as usize;

    {
        let mut view = img.as_opencv_mut()?;
        view.data_typed_mut::<u8>()?[0] = 9;
    }
    assert_eq!(img.buffer()[0], 9);

    {
        let exclusive: &mut Image<u8, 1> = &mut img;
        let mut view = exclusive.as_opencv_mut()?;
        view.data_typed_mut::<u8>()?[1] = 8;
    }
    assert_eq!(img.buffer()[1], 8);
    assert_eq!(img.buffer().as_ptr() as usize, ptr);
    Ok(())
}

#[test]
fn as_opencv_mut_writes_through_for_dynamic_image() -> TestResult {
    let mut dynamic = DynamicImage::from(Image::<u8, 1>::new_vec(vec![1, 2, 3, 4], NZ_2, NZ_2)?);

    {
        let mut view = dynamic.as_opencv_mut()?;
        view.data_typed_mut::<u8>()?[0] = 9;
    }

    let channel = dynamic.first().try_cast::<u8>()?;
    assert_eq!(channel.buffer_flat()[0], 9);
    Ok(())
}

#[test]
fn opencv_mat_into_dynamic_imbuf_shares_buffer() -> TestResult {
    let mat = Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]])?;
    let ptr = mat.data() as usize;

    let dynamic = mat.into_imbuf_dynamic()?;
    let channel = dynamic.first().try_cast::<u8>()?;
    assert_eq!(channel.buffer_flat().as_ptr() as usize, ptr);
    assert_eq!(channel.dimensions(), dims(3, 2)?);
    Ok(())
}

#[test]
fn opencv_mat_into_specific_imbuf() -> TestResult {
    let mat = Mat::from_slice_2d(&[[1u16, 2], [3, 4], [5, 6]])?;
    let image: Image<u16, 1> = mat.into_imbuf_specific()?;
    assert_eq!(image.dimensions(), dims(2, 3)?);
    assert_eq!(image.buffer(), &[1, 2, 3, 4, 5, 6]);
    Ok(())
}
