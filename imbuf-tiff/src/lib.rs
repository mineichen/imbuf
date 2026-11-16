#![doc = include_str!("../README.md")]

use std::{
    io::{self, Read, Seek},
    num::{NonZeroU8, NonZeroU32},
};

use imbuf::{DynamicImage, DynamicImageChannel, PixelTypePrimitive};
use tiff::{
    ColorType, TiffError,
    decoder::{Decoder, DecodingResult},
    tags::{PhotometricInterpretation, PlanarConfiguration, Tag},
};

/// Decodes a TIFF from `data` into an [`imbuf::DynamicImage`] with its raw samples.
///
/// - Each page (image file directory) becomes a channel of the [`imbuf::DynamicImage`], in file
///   order. Pages can have different sample types and dimensions, e.g. an `f32` page followed by
///   a smaller `u16` page.
/// - Previews (reduced resolution pages, e.g. thumbnails) are skipped, wherever they are.
/// - Transparency mask pages are loaded like image pages, if they have a supported sample type.
///   1 bit transparency masks are not supported.
/// - Multi-page documents (a page is flagged as page of a multi-page document and there are
///   several full resolution pages) are rejected.
/// - The samples of a page are the elements of its pixels, e.g. `u16` RGB becomes `[u16; 3]`
///   and a 5-band `u8` page becomes `[u8; 5]`.
/// - Samples are loaded as-is, which includes `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`,
///   `i64`, `f32` and `f64`, e.g. a grayscale `u32` TIFF becomes `imbuf::Image<u32, 1>`.
///
/// # Errors
/// - [`io::ErrorKind::Unsupported`], if `data` is not a TIFF, if it is a multi-page document or
///   if a page can't be loaded as raw samples (e.g. CMYK, YCbCr, palette, planar or 1 bit pages)
/// - [`io::ErrorKind::InvalidData`], if the TIFF is malformed, has a zero dimension or has no
///   full resolution image page
/// - [`io::ErrorKind::OutOfMemory`], if the decoding limits of `tiff` are exceeded
pub fn load_from_memory(data: &[u8]) -> io::Result<DynamicImage> {
    if !is_tiff(data) {
        return Err(io::Error::new(io::ErrorKind::Unsupported, "Not a TIFF"));
    }
    load(data).map_err(|Error(error)| error)
}

/// Checks the byte order mark and the magic number of TIFF and BigTIFF
fn is_tiff(data: &[u8]) -> bool {
    matches!(
        data,
        [b'I', b'I', 42 | 43, 0, ..] | [b'M', b'M', 0, 42 | 43, ..]
    )
}

/// [`io::Error`], which `?` can convert [`TiffError`]s into
struct Error(io::Error);

impl Error {
    fn unsupported(feature: String) -> Self {
        Self(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("TIFF feature is not supported: {feature}"),
        ))
    }

    fn invalid_data(message: &str) -> Self {
        Self(io::Error::new(io::ErrorKind::InvalidData, message))
    }

    fn unsupported_page(page: usize) -> Self {
        Self::unsupported(format!(
            "of page {page} (only chunky Gray/GrayA/RGB/RGBA/Multiband pages with 8, 16, 32 or 64 \
             bit integer or 32/64 bit float samples can be loaded as raw samples, 1 bit \
             transparency masks are not supported)"
        ))
    }

    fn zero_dimension() -> Self {
        Self::invalid_data("Neither width nor height can be 0")
    }

    fn no_image_page() -> Self {
        Self::invalid_data("No full resolution image page (only previews or transparency masks)")
    }
}

impl From<TiffError> for Error {
    fn from(error: TiffError) -> Self {
        Self(match error {
            TiffError::IoError(error) => error,
            TiffError::UnsupportedError(error) => return Self::unsupported(error.to_string()),
            TiffError::LimitsExceeded => io::Error::new(io::ErrorKind::OutOfMemory, error),
            error => io::Error::new(io::ErrorKind::InvalidData, error),
        })
    }
}

/// Bits of `NewSubfileType`
const NEW_SUBFILE_REDUCED_RESOLUTION: u32 = 1;
const NEW_SUBFILE_MULTI_PAGE: u32 = 1 << 1;
const NEW_SUBFILE_TRANSPARENCY_MASK: u32 = 1 << 2;
/// Values of the deprecated `SubfileType`
const SUBFILE_REDUCED_RESOLUTION: u16 = 2;
const SUBFILE_MULTI_PAGE: u16 = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PageKind {
    /// Reduced resolution version of another page, e.g. a thumbnail
    Preview,
    TransparencyMask,
    /// Full resolution image page
    Image {
        multi_page: bool,
    },
}

fn page_kind<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<PageKind, Error> {
    let new_subfile_type = decoder
        .find_tag_unsigned::<u32>(Tag::NewSubfileType)?
        .unwrap_or(0);
    let subfile_type = decoder.find_tag_unsigned::<u16>(Tag::SubfileType)?;
    let photometric = decoder.find_tag_unsigned::<u16>(Tag::PhotometricInterpretation)?;

    Ok(
        if new_subfile_type & NEW_SUBFILE_REDUCED_RESOLUTION != 0
            || subfile_type == Some(SUBFILE_REDUCED_RESOLUTION)
        {
            PageKind::Preview
        } else if new_subfile_type & NEW_SUBFILE_TRANSPARENCY_MASK != 0
            || photometric == Some(PhotometricInterpretation::TransparencyMask.to_u16())
        {
            PageKind::TransparencyMask
        } else {
            PageKind::Image {
                multi_page: new_subfile_type & NEW_SUBFILE_MULTI_PAGE != 0
                    || subfile_type == Some(SUBFILE_MULTI_PAGE),
            }
        },
    )
}

fn load(data: &[u8]) -> Result<DynamicImage, Error> {
    let mut decoder = Decoder::new(io::Cursor::new(data))?;
    let mut channels = Vec::new();
    let mut image_pages = 0_usize;
    let mut multi_page = false;
    // First page which can't be loaded as raw samples
    let mut unsupported_page = None;

    for page in 0.. {
        if page > 0 {
            if !decoder.more_images() {
                break;
            }
            decoder.next_image()?;
        }
        let kind = page_kind(&mut decoder)?;
        if let PageKind::Image { multi_page: flag } = kind {
            image_pages += 1;
            multi_page |= flag;
            if multi_page && image_pages > 1 {
                return Err(Error::unsupported("multi-page documents".into()));
            }
        }
        // Only collect metadata, once the TIFF is known to be unsupported
        if kind == PageKind::Preview || unsupported_page.is_some() {
            continue;
        }
        // 1 bit masks (`PhotometricInterpretation` 4) aren't supported (yet), as `imbuf` has
        // no 1 bit primitive. Masks with other layouts are loaded like image pages
        let bit_mask = kind == PageKind::TransparencyMask
            && decoder.find_tag_unsigned::<u16>(Tag::PhotometricInterpretation)?
                == Some(PhotometricInterpretation::TransparencyMask.to_u16());
        let channel = if bit_mask {
            None
        } else {
            load_page(&mut decoder)?
        };
        match channel {
            Some(channel) => channels.push(channel),
            None => unsupported_page = Some(page),
        }
    }

    if image_pages == 0 {
        return Err(Error::no_image_page());
    }
    if let Some(page) = unsupported_page {
        return Err(Error::unsupported_page(page));
    }
    let mut channels = channels.into_iter();
    let first = channels.next().ok_or_else(Error::no_image_page)?;
    Ok(DynamicImage::from_channels(first, channels))
}

/// Loads the current page as a single channel with the samples as pixel elements.
/// Returns `None`, if the samples can't be represented as-is
fn load_page<R: Read + Seek>(
    decoder: &mut Decoder<R>,
) -> Result<Option<DynamicImageChannel>, Error> {
    let (bits, samples) = match decoder.colortype()? {
        ColorType::Gray(bits) => (bits, 1),
        ColorType::GrayA(bits) => (bits, 2),
        ColorType::RGB(bits) => (bits, 3),
        ColorType::RGBA(bits) => (bits, 4),
        ColorType::Multiband {
            bit_depth,
            num_samples,
        } => (bit_depth, num_samples),
        // Require a conversion (e.g. to RGB) to be meaningful, or are unknown
        ColorType::Palette(_)
        | ColorType::CMYK(_)
        | ColorType::CMYKA(_)
        | ColorType::YCbCr(_)
        | _ => return Ok(None),
    };
    // Smaller samples are packed into bytes by `tiff`
    if !matches!(bits, 8 | 16 | 32 | 64) {
        return Ok(None);
    }
    let Some(pixel_elements) = u8::try_from(samples).ok().and_then(NonZeroU8::new) else {
        return Ok(None);
    };
    // `tiff` only decodes the first plane of planar pages
    let planar = decoder.find_tag_unsigned::<u16>(Tag::PlanarConfiguration)?;
    if samples > 1 && planar == Some(PlanarConfiguration::Planar.to_u16()) {
        return Ok(None);
    }
    let (width, height) = decoder.dimensions()?;
    let (Some(width), Some(height)) = (NonZeroU32::new(width), NonZeroU32::new(height)) else {
        return Err(Error::zero_dimension());
    };

    let channel = match decoder.read_image()? {
        DecodingResult::U8(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::U16(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::U32(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::U64(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::I8(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::I16(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::I32(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::I64(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::F32(buffer) => new_channel(buffer, width, height, pixel_elements),
        DecodingResult::F64(buffer) => new_channel(buffer, width, height, pixel_elements),
        // imbuf has no f16 primitive
        DecodingResult::F16(_) => return Ok(None),
    };
    Ok(Some(channel))
}

fn new_channel<T: PixelTypePrimitive + bytemuck::Pod>(
    buffer: Vec<T>,
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
) -> DynamicImageChannel {
    DynamicImageChannel::new_vec_flat(buffer, width, height, pixel_elements)
        .expect("tiff decodes width * height * samples samples for chunky pages")
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, Write};

    use imbuf::{DynamicImageChannel, Image, ImageChannel};
    use testresult::TestResult;
    use tiff::{
        encoder::{TiffEncoder, TiffKind, colortype},
        tags::{PhotometricInterpretation, SampleFormat, Tag},
    };

    use super::*;

    /// 5 `u8` samples per pixel, for which `tiff` has no predefined color type
    struct Multiband5;
    impl colortype::ColorType for Multiband5 {
        type Inner = u8;
        const TIFF_VALUE: PhotometricInterpretation = PhotometricInterpretation::BlackIsZero;
        const BITS_PER_SAMPLE: &'static [u16] = &[8; 5];
        const SAMPLE_FORMAT: &'static [SampleFormat] = &[SampleFormat::Uint; 5];

        fn horizontal_predict(row: &[u8], result: &mut Vec<u8>) {
            result.extend_from_slice(row);
        }
    }

    fn encode<C: colortype::ColorType>(
        width: u32,
        height: u32,
        data: &[C::Inner],
    ) -> TestResult<Vec<u8>>
    where
        [C::Inner]: tiff::encoder::TiffValue,
    {
        let mut tiff = Cursor::new(Vec::new());
        TiffEncoder::new(&mut tiff)?.write_image::<C>(width, height, data)?;
        Ok(tiff.into_inner())
    }

    /// Appends a page with `NewSubfileType` (e.g. 1 for a reduced resolution preview), if not 0
    fn write_page<C: colortype::ColorType, W: Write + Seek, K: TiffKind>(
        encoder: &mut TiffEncoder<W, K>,
        width: u32,
        height: u32,
        data: &[C::Inner],
        new_subfile_type: u32,
    ) -> TestResult
    where
        [C::Inner]: tiff::encoder::TiffValue,
    {
        let mut image = encoder.new_image::<C>(width, height)?;
        if new_subfile_type != 0 {
            image
                .encoder()
                .write_tag(Tag::NewSubfileType, new_subfile_type)?;
        }
        image.write_data(data)?;
        Ok(())
    }

    fn channel_dimensions(channel: &DynamicImageChannel) -> (u32, u32) {
        let (width, height) = channel.dimensions();
        (width.get(), height.get())
    }

    fn assert_error(result: io::Result<DynamicImage>, kind: io::ErrorKind, message: &str) {
        match result {
            Ok(_) => panic!("Expected {kind:?} error containing {message:?}, got an image"),
            Err(error) => {
                assert_eq!(error.kind(), kind, "{error}");
                assert!(error.to_string().contains(message), "{error}");
            }
        }
    }

    #[test]
    fn load_non_tiff_is_unsupported() {
        // PNG signature
        let png = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
        assert_error(
            load_from_memory(&png),
            io::ErrorKind::Unsupported,
            "Not a TIFF",
        );
        assert_error(
            load_from_memory(&[]),
            io::ErrorKind::Unsupported,
            "Not a TIFF",
        );
        assert_error(
            load_from_memory(b"II"),
            io::ErrorKind::Unsupported,
            "Not a TIFF",
        );
    }

    #[test]
    fn load_truncated_tiff_fails() -> TestResult {
        let tiff = encode::<colortype::Gray8>(2, 1, &[1, 2])?;
        let error = load_from_memory(&tiff[..10])
            .err()
            .ok_or("expected an error")?;
        assert_ne!(error.kind(), io::ErrorKind::Unsupported, "{error}");
        Ok(())
    }

    #[test]
    fn load_tiff_luma_f32() -> TestResult {
        let buf = [0., 0.5, 1., -1., 1e10, 0.25];
        let tiff = encode::<colortype::Gray32Float>(3, 2, &buf)?;

        let image = load_from_memory(&tiff)?;
        let image = Image::<f32, 1>::try_from(image)?;
        assert_eq!(image.dimensions().0.get(), 3);
        assert_eq!(image.dimensions().1.get(), 2);
        assert_eq!(image.buffer(), &buf);
        Ok(())
    }

    #[test]
    fn load_tiff_luma_u32() -> TestResult {
        let buf = [0, 1, u32::MAX, 1 << 20];
        let tiff = encode::<colortype::Gray32>(2, 2, &buf)?;

        let image = Image::<u32, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &buf);
        Ok(())
    }

    #[test]
    fn load_tiff_luma_other_primitives() -> TestResult {
        let tiff = encode::<colortype::Gray64>(2, 1, &[0, u64::MAX])?;
        let image = Image::<u64, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[0, u64::MAX]);

        let tiff = encode::<colortype::GrayI8>(2, 1, &[-1, i8::MIN])?;
        let image = Image::<i8, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[-1, i8::MIN]);

        let tiff = encode::<colortype::GrayI16>(2, 1, &[-1, i16::MIN])?;
        let image = Image::<i16, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[-1, i16::MIN]);

        let tiff = encode::<colortype::GrayI32>(2, 1, &[-1, i32::MIN])?;
        let image = Image::<i32, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[-1, i32::MIN]);

        let tiff = encode::<colortype::GrayI64>(2, 1, &[-1, i64::MIN])?;
        let image = Image::<i64, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[-1, i64::MIN]);

        let tiff = encode::<colortype::Gray64Float>(2, 1, &[-0.5, 1e300])?;
        let image = Image::<f64, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[-0.5, 1e300]);
        Ok(())
    }

    #[test]
    fn load_tiff_luma8() -> TestResult {
        let tiff = encode::<colortype::Gray8>(2, 1, &[1, 2])?;

        let image = Image::<u8, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn load_tiff_luma16() -> TestResult {
        let tiff = encode::<colortype::Gray16>(2, 1, &[1, 2])?;

        let image = Image::<u16, 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn load_tiff_rgb_f32_interleaved() -> TestResult {
        let tiff = encode::<colortype::RGB32Float>(1, 1, &[0.25, 0.5, 1.])?;

        let image = Image::<[f32; 3], 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[[0.25, 0.5, 1.]]);
        Ok(())
    }

    #[test]
    fn load_tiff_rgba_u32_interleaved() -> TestResult {
        let tiff = encode::<colortype::RGBA32>(1, 1, &[1, 2, 3, u32::MAX])?;

        let image = Image::<[u32; 4], 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[[1, 2, 3, u32::MAX]]);
        Ok(())
    }

    #[test]
    fn load_tiff_5_samples_interleaved() -> TestResult {
        let buf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let tiff = encode::<Multiband5>(2, 1, &buf)?;

        let image = Image::<[u8; 5], 1>::try_from(load_from_memory(&tiff)?)?;
        assert_eq!(image.buffer(), &[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]);
        Ok(())
    }

    #[test]
    fn load_tiff_5_pages_as_5_channels() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        for i in 0..5_u16 {
            write_page::<colortype::Gray16, _, _>(&mut encoder, 2, 1, &[i, i * 10], 0)?;
        }

        let image = Image::<u16, 5>::try_from(load_from_memory(tiff.get_ref())?)?;
        for (i, buffer) in (0..5_u16).zip(image.buffers()) {
            assert_eq!(buffer, &[i, i * 10]);
        }
        Ok(())
    }

    #[test]
    fn load_tiff_3_pages_with_different_types_and_sizes() -> TestResult {
        let first = [0.5, -1., 2., 3., 4., 1e10];
        let second = [1, u16::MAX];
        let third = [0.25];
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray32Float, _, _>(&mut encoder, 3, 2, &first, 0)?;
        write_page::<colortype::Gray16, _, _>(&mut encoder, 1, 2, &second, 0)?;
        write_page::<colortype::Gray32Float, _, _>(&mut encoder, 1, 1, &third, 0)?;

        let image = load_from_memory(tiff.get_ref())?;
        assert_eq!(image.len(), 3);
        assert!(!image.is_uniform());
        let [first_channel, second_channel, third_channel] = &image[..] else {
            unreachable!()
        };

        assert_eq!(channel_dimensions(first_channel), (3, 2));
        assert_eq!(first_channel.buffer::<f32>()?, &first);
        assert_eq!(channel_dimensions(second_channel), (1, 2));
        assert_eq!(second_channel.buffer::<u16>()?, &second);
        assert_eq!(channel_dimensions(third_channel), (1, 1));
        assert_eq!(third_channel.buffer::<f32>()?, &third);

        let second_channel = ImageChannel::<u16>::try_from(second_channel.clone())?;
        assert_eq!(second_channel.buffer(), &second);
        Ok(())
    }

    /// Appends an uncompressed 1 bit transparency mask page with a single strip
    fn write_bit_mask_page<W: Write + Seek, K: TiffKind>(
        encoder: &mut TiffEncoder<W, K>,
        width: u32,
        height: u32,
        strip: &[u8],
    ) -> TestResult {
        let mut directory = encoder.image_directory()?;
        let offset = directory.write_data(strip)?;
        directory.write_tag(Tag::NewSubfileType, 4_u32)?;
        directory.write_tag(Tag::ImageWidth, width)?;
        directory.write_tag(Tag::ImageLength, height)?;
        directory.write_tag(Tag::BitsPerSample, 1_u16)?;
        directory.write_tag(Tag::Compression, 1_u16)?;
        directory.write_tag(
            Tag::PhotometricInterpretation,
            PhotometricInterpretation::TransparencyMask.to_u16(),
        )?;
        directory.write_tag(Tag::StripOffsets, u32::try_from(offset)?)?;
        directory.write_tag(Tag::RowsPerStrip, height)?;
        directory.write_tag(Tag::StripByteCounts, u32::try_from(strip.len())?)?;
        directory.finish()?;
        Ok(())
    }

    #[test]
    fn load_tiff_skips_previews() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        // First page preview, as e.g. in DNG
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[10], 1)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 2, 2, &[1, 2, 3, 4], 0)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[5], 1)?;
        // Preview flagged by the deprecated `SubfileType`
        let mut preview = encoder.new_image::<colortype::Gray8>(1, 1)?;
        preview.encoder().write_tag(Tag::SubfileType, 2_u16)?;
        preview.write_data(&[11])?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 2, 2, &[6, 7, 8, 9], 0)?;

        let image = Image::<u8, 2>::try_from(load_from_memory(tiff.get_ref())?)?;
        let [first, second] = image.buffers();
        assert_eq!(first, &[1, 2, 3, 4]);
        assert_eq!(second, &[6, 7, 8, 9]);
        Ok(())
    }

    #[test]
    fn load_tiff_only_previews_fails() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[1], 1)?;

        assert_error(
            load_from_memory(tiff.get_ref()),
            io::ErrorKind::InvalidData,
            "No full resolution image page",
        );
        Ok(())
    }

    #[test]
    fn load_tiff_cmyk_is_unsupported() -> TestResult {
        let tiff = encode::<colortype::CMYK8>(1, 1, &[0, 0, 0, 0])?;

        assert_error(
            load_from_memory(&tiff),
            io::ErrorKind::Unsupported,
            "of page 0 ",
        );
        Ok(())
    }

    #[test]
    fn load_tiff_cmyk_with_preview_is_unsupported() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::CMYK8, _, _>(&mut encoder, 1, 1, &[0, 0, 0, 0], 0)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[1], 1)?;

        assert_error(
            load_from_memory(tiff.get_ref()),
            io::ErrorKind::Unsupported,
            "of page 0 ",
        );
        Ok(())
    }

    #[test]
    fn load_tiff_multi_page_document_fails() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[1], 2)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[2], 2)?;

        assert_error(
            load_from_memory(tiff.get_ref()),
            io::ErrorKind::Unsupported,
            "multi-page documents",
        );
        Ok(())
    }

    #[test]
    fn load_tiff_multi_page_document_flagged_by_deprecated_tag_fails() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[1], 0)?;
        let mut page = encoder.new_image::<colortype::Gray8>(1, 1)?;
        page.encoder().write_tag(Tag::SubfileType, 3_u16)?;
        page.write_data(&[2])?;

        assert_error(
            load_from_memory(tiff.get_ref()),
            io::ErrorKind::Unsupported,
            "multi-page documents",
        );
        Ok(())
    }

    #[test]
    fn load_tiff_single_page_flagged_as_multi_page_document() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[1], 2)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[2], 1)?;

        let image = Image::<u8, 1>::try_from(load_from_memory(tiff.get_ref())?)?;
        assert_eq!(image.buffer(), &[1]);
        Ok(())
    }

    #[test]
    fn load_tiff_bit_transparency_mask_fails() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 8, 1, &[1; 8], 0)?;
        write_bit_mask_page(&mut encoder, 8, 1, &[0b1010_1010])?;

        assert_error(
            load_from_memory(tiff.get_ref()),
            io::ErrorKind::Unsupported,
            "of page 1 ",
        );
        Ok(())
    }

    #[test]
    fn load_tiff_8_bit_transparency_mask() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray16, _, _>(&mut encoder, 2, 1, &[1, 2], 0)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 2, 1, &[0, 255], 4)?;

        let image = load_from_memory(tiff.get_ref())?;
        let [pixel_channel, mask_channel] = &image[..] else {
            panic!("Expected 2 channels, got {}", image.len());
        };
        assert_eq!(pixel_channel.buffer::<u16>()?, &[1, 2]);
        assert_eq!(mask_channel.buffer::<u8>()?, &[0, 255]);
        Ok(())
    }

    #[test]
    fn load_tiff_multi_page_with_cmyk_page_fails() -> TestResult {
        let mut tiff = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut tiff)?;
        write_page::<colortype::Gray8, _, _>(&mut encoder, 1, 1, &[1], 0)?;
        write_page::<colortype::CMYK8, _, _>(&mut encoder, 1, 1, &[0, 0, 0, 0], 0)?;

        assert_error(
            load_from_memory(tiff.get_ref()),
            io::ErrorKind::Unsupported,
            "of page 1 ",
        );
        Ok(())
    }
}
