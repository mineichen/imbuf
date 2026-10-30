use assert_matches::assert_matches;
use std::num::NonZeroU32;

use imbuf::{
    DynamicImage, Image, ImageChannel, ImageView, IncompatibleImage, IncompatibleImageError,
};
use testresult::TestResult;

const NZ_1: NonZeroU32 = NonZeroU32::MIN;
const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

#[test]
fn create_from_rgbf32() -> TestResult {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], NZ_1, NZ_1)?;
    let dynamic = DynamicImage::from(image.clone());
    let back: Image<f32, 3> = dynamic.try_into()?;
    assert_eq!(back, image);
    Ok(())
}
#[test]
fn create_from_dynamic_with_different_size() -> TestResult {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], NZ_1, NZ_1)?;
    let mut dynamic = DynamicImage::from(image);
    dynamic[0] = ImageChannel::new_vec(vec![42f32, 42.], NZ_1, NZ_2)?.into();

    assert_matches!(
        Result::<Image<f32, 3>, IncompatibleImage<_>>::from(dynamic.clone()),
        Err(IncompatibleImage { image, .. }) if image == dynamic
    );
    assert_matches!(
        <Image<f32, 3>>::try_from(dynamic),
        Err(IncompatibleImageError::MixedImageSizes { .. })
    );
    Ok(())
}

#[test]
fn create_from_dynamic_with_different_type() -> TestResult {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], NZ_1, NZ_1)?;
    let mut dynamic = DynamicImage::from(image);
    dynamic[0] = ImageChannel::new_vec(vec![1u16], NZ_1, NZ_1)?.into();

    assert_matches!(
        Result::<Image<f32, 3>, IncompatibleImage<_>>::from(dynamic.clone()),
        Err(IncompatibleImage { image, .. }) if image == dynamic
    );
    assert_matches!(
        <Image<f32, 3>>::try_from(dynamic),
        Err(IncompatibleImageError::Channel(_))
    );
    Ok(())
}

#[test]
fn create_from_dynamic_with_too_few_channels() -> TestResult {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], NZ_1, NZ_1)?;
    let dynamic = DynamicImage::from(image);
    assert_matches!(
        Result::<Image<f32, 4>, IncompatibleImage<_>>::from(dynamic.clone()),
        Err(IncompatibleImage { image, .. }) if image == dynamic
    );
    assert_matches!(
        <Image<f32, 4>>::try_from(dynamic),
        Err(IncompatibleImageError::ChannelCount(_))
    );
    Ok(())
}

#[test]
fn create_from_dynamic_with_too_many_channels() -> TestResult {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], NZ_1, NZ_1)?;
    let dynamic = DynamicImage::from(image);
    assert_matches!(
        Result::<Image<f32, 2>, IncompatibleImage<_>>::from(dynamic.clone()),
        Err(IncompatibleImage { image, .. }) if image == dynamic
    );
    assert_matches!(<ImageView<f32, 2>>::try_from(&dynamic), Err(_));
    Ok(())
}
