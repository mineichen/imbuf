use std::ops::{Deref, DerefMut};

use image::{ImageBuffer, Luma, LumaA, Rgb, Rgba};

/// A borrowed view of an [`imbuf`] image as an [`image`] `DynamicImage`-shaped
/// value.
///
/// Unlike [`image::DynamicImage`], this type can borrow the buffer, so
/// converting from an `imbuf` image never copies.
#[derive(Debug)]
pub enum DynamicRefImage<'a> {
    ImageLuma8(ImageBuffer<Luma<u8>, &'a [u8]>),
    ImageLuma16(ImageBuffer<Luma<u16>, &'a [u16]>),
    ImageLumaA8(ImageBuffer<LumaA<u8>, &'a [u8]>),
    ImageLumaA16(ImageBuffer<LumaA<u16>, &'a [u16]>),
    ImageRgb8(ImageBuffer<Rgb<u8>, &'a [u8]>),
    ImageRgb16(ImageBuffer<Rgb<u16>, &'a [u16]>),
    ImageRgb32F(ImageBuffer<Rgb<f32>, &'a [f32]>),
    ImageRgba8(ImageBuffer<Rgba<u8>, &'a [u8]>),
    ImageRgba16(ImageBuffer<Rgba<u16>, &'a [u16]>),
    ImageRgba32F(ImageBuffer<Rgba<f32>, &'a [f32]>),
}

/// A mutable borrowed view of an [`imbuf`] image as an [`image`]
/// `DynamicImage`-shaped value.
#[derive(Debug)]
pub enum DynamicRefImageMut<'a> {
    ImageLuma8(ImageBuffer<Luma<u8>, &'a mut [u8]>),
    ImageLuma16(ImageBuffer<Luma<u16>, &'a mut [u16]>),
    ImageLumaA8(ImageBuffer<LumaA<u8>, &'a mut [u8]>),
    ImageLumaA16(ImageBuffer<LumaA<u16>, &'a mut [u16]>),
    ImageRgb8(ImageBuffer<Rgb<u8>, &'a mut [u8]>),
    ImageRgb16(ImageBuffer<Rgb<u16>, &'a mut [u16]>),
    ImageRgb32F(ImageBuffer<Rgb<f32>, &'a mut [f32]>),
    ImageRgba8(ImageBuffer<Rgba<u8>, &'a mut [u8]>),
    ImageRgba16(ImageBuffer<Rgba<u16>, &'a mut [u16]>),
    ImageRgba32F(ImageBuffer<Rgba<f32>, &'a mut [f32]>),
}

macro_rules! impl_from_borrowed_image_buffer {
    ($color:ty, $primitive:ty, $variant:ident) => {
        impl<'a, 'b: 'a> From<&'a ImageBuffer<$color, Vec<$primitive>>> for DynamicRefImage<'a> {
            fn from(buffer: &'a ImageBuffer<$color, Vec<$primitive>>) -> Self {
                let data: &'a [$primitive] = buffer.deref();
                DynamicRefImage::$variant(
                    ImageBuffer::from_raw(buffer.width(), buffer.height(), data)
                        .expect("Incompatible buffer size"),
                )
            }
        }

        impl<'a, 'b: 'a> From<&'a mut ImageBuffer<$color, Vec<$primitive>>>
            for DynamicRefImageMut<'a>
        {
            fn from(buffer: &'a mut ImageBuffer<$color, Vec<$primitive>>) -> Self {
                let width = buffer.width();
                let height = buffer.height();
                let data: &'a mut [$primitive] = buffer.deref_mut();

                DynamicRefImageMut::$variant(
                    ImageBuffer::from_raw(width, height, data).expect("Incompatible buffer size"),
                )
            }
        }
    };
}

impl_from_borrowed_image_buffer!(Luma<u8>, u8, ImageLuma8);
impl_from_borrowed_image_buffer!(Luma<u16>, u16, ImageLuma16);
impl_from_borrowed_image_buffer!(LumaA<u8>, u8, ImageLumaA8);
impl_from_borrowed_image_buffer!(LumaA<u16>, u16, ImageLumaA16);
impl_from_borrowed_image_buffer!(Rgb<u8>, u8, ImageRgb8);
impl_from_borrowed_image_buffer!(Rgb<u16>, u16, ImageRgb16);
impl_from_borrowed_image_buffer!(Rgb<f32>, f32, ImageRgb32F);
impl_from_borrowed_image_buffer!(Rgba<u8>, u8, ImageRgba8);
impl_from_borrowed_image_buffer!(Rgba<u16>, u16, ImageRgba16);
impl_from_borrowed_image_buffer!(Rgba<f32>, f32, ImageRgba32F);

impl DynamicRefImage<'_> {
    /// # Errors
    /// Forwards errors of `image::write_to`
    pub fn write_to<W: std::io::Write + std::io::Seek>(
        &self,
        mut buffer: W,
        format: image::ImageFormat,
    ) -> Result<(), image::ImageError> {
        match self {
            DynamicRefImage::ImageLuma8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageLuma16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageLumaA8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageLumaA16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgb8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgb16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgb32F(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgba8(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgba16(x) => x.write_to(&mut buffer, format),
            DynamicRefImage::ImageRgba32F(x) => x.write_to(&mut buffer, format),
        }
    }
}

impl DynamicRefImageMut<'_> {
    /// # Errors
    /// Forwards errors of `image::write_to`
    pub fn write_to<W: std::io::Write + std::io::Seek>(
        &self,
        mut buffer: W,
        format: image::ImageFormat,
    ) -> Result<(), image::ImageError> {
        match self {
            DynamicRefImageMut::ImageLuma8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageLuma16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageLumaA8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageLumaA16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgb8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgb16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgb32F(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgba8(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgba16(x) => x.write_to(&mut buffer, format),
            DynamicRefImageMut::ImageRgba32F(x) => x.write_to(&mut buffer, format),
        }
    }
}
