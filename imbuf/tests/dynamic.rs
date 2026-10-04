use std::num::NonZeroU32;

use imbuf::{DynamicImage, Image, ImageChannel};

const ONE: NonZeroU32 = NonZeroU32::MIN;
const TWO: NonZeroU32 = NonZeroU32::new(2).unwrap();

#[test]
fn create_from_rgbf32() {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], ONE, ONE);
    let dynamic = DynamicImage::from(image.clone());
    let back: Image<f32, 3> = dynamic.try_into().unwrap();
    assert_eq!(back, image);
}
#[test]
fn create_from_dynamic_with_different_size() {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], ONE, ONE);
    let mut dynamic = DynamicImage::from(image);
    dynamic[0] = ImageChannel::new_vec(vec![42f32, 42.], ONE, TWO).into();

    let back = <Image<f32, 3>>::try_from(dynamic.clone()).unwrap_err();
    assert_eq!(back.image, dynamic);
}

#[test]
fn create_from_dynamic_with_different_type() {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], ONE, ONE);
    let mut dynamic = DynamicImage::from(image);
    dynamic[0] = ImageChannel::new_vec(vec![1u16], ONE, ONE).into();

    let back = <Image<f32, 3>>::try_from(dynamic.clone()).unwrap_err();
    assert_eq!(back.image, dynamic);
}

#[test]
fn create_from_dynamic_with_too_few_channels() {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], ONE, ONE);
    let dynamic = DynamicImage::from(image);
    let back = <Image<f32, 4>>::try_from(dynamic.clone()).unwrap_err();
    assert_eq!(back.image, dynamic);
}

#[test]
/// This should be allowed for taking a RGB from a dynamic RGBA
fn create_from_dynamic_with_too_many_channels() {
    let image: Image<f32, 3> = Image::new_vec(vec![42f32, 42., 43.], ONE, ONE);
    let dynamic = DynamicImage::from(image);
    let back: Image<f32, 2> = dynamic.try_into().unwrap();
    assert_eq!(back, Image::new_vec(vec![42f32, 42.], ONE, ONE));
}
