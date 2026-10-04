use std::num::NonZeroU8;

use crate::{
    ImageChannel,
    dynamic::DynamicImageChannel,
    pixel_elements::{ComptimeSize, PixelSize, RuntimeSize},
};

/// Removes all compile time hints, of how many channels a pixel persists
/// This is primarily used in `DynamicImageChannel`
#[derive(Clone, Copy, Default)]
pub struct DynamicSize<T: PixelTypePrimitive>(std::marker::PhantomData<T>);

impl<T: PixelTypePrimitive> RuntimePixelType for DynamicSize<T> {
    type Primitive = T;
    type PixelSize = RuntimeSize;
}

pub trait PixelTypePrimitive:
    Clone + PartialEq + Send + Sync + 'static + crate::seal::SealedPrimitive
{
    fn into_runtime_channel(i: ImageChannel<DynamicSize<Self>>) -> DynamicImageChannel;

    /// # Errors
    /// Returns `Err` if the dynamic image channel is not compatible with the pixel type.
    fn try_from_dynamic_image(
        channel: DynamicImageChannel,
    ) -> Result<ImageChannel<DynamicSize<Self>>, DynamicImageChannel>;

    fn try_from_dynamic_image_ref(
        channel: &DynamicImageChannel,
    ) -> Option<&ImageChannel<DynamicSize<Self>>>;
    fn try_from_dynamic_image_mut(
        channel: &mut DynamicImageChannel,
    ) -> Option<&mut ImageChannel<DynamicSize<Self>>>;
}

impl PixelTypePrimitive for u8 {
    fn into_runtime_channel(i: ImageChannel<DynamicSize<Self>>) -> DynamicImageChannel {
        DynamicImageChannel::U8(i)
    }

    fn try_from_dynamic_image(
        channel: DynamicImageChannel,
    ) -> Result<ImageChannel<DynamicSize<Self>>, DynamicImageChannel> {
        if let DynamicImageChannel::U8(channel) = channel {
            Ok(channel)
        } else {
            Err(channel)
        }
    }

    fn try_from_dynamic_image_ref(
        channel: &DynamicImageChannel,
    ) -> Option<&ImageChannel<DynamicSize<Self>>> {
        if let DynamicImageChannel::U8(channel) = channel {
            Some(channel)
        } else {
            None
        }
    }

    fn try_from_dynamic_image_mut(
        channel: &mut DynamicImageChannel,
    ) -> Option<&mut ImageChannel<DynamicSize<Self>>> {
        if let DynamicImageChannel::U8(channel) = channel {
            Some(channel)
        } else {
            None
        }
    }
}
impl PixelTypePrimitive for u16 {
    fn into_runtime_channel(i: ImageChannel<DynamicSize<Self>>) -> DynamicImageChannel {
        DynamicImageChannel::U16(i)
    }
    fn try_from_dynamic_image(
        channel: DynamicImageChannel,
    ) -> Result<ImageChannel<DynamicSize<Self>>, DynamicImageChannel> {
        if let DynamicImageChannel::U16(channel) = channel {
            Ok(channel)
        } else {
            Err(channel)
        }
    }

    fn try_from_dynamic_image_ref(
        channel: &DynamicImageChannel,
    ) -> Option<&ImageChannel<DynamicSize<Self>>> {
        if let DynamicImageChannel::U16(channel) = channel {
            Some(channel)
        } else {
            None
        }
    }

    fn try_from_dynamic_image_mut(
        channel: &mut DynamicImageChannel,
    ) -> Option<&mut ImageChannel<DynamicSize<Self>>> {
        if let DynamicImageChannel::U16(channel) = channel {
            Some(channel)
        } else {
            None
        }
    }
}

impl PixelTypePrimitive for f32 {
    fn into_runtime_channel(i: ImageChannel<DynamicSize<Self>>) -> DynamicImageChannel {
        DynamicImageChannel::F32(i)
    }
    fn try_from_dynamic_image(
        channel: DynamicImageChannel,
    ) -> Result<ImageChannel<DynamicSize<Self>>, DynamicImageChannel> {
        if let DynamicImageChannel::F32(channel) = channel {
            Ok(channel)
        } else {
            Err(channel)
        }
    }

    fn try_from_dynamic_image_ref(
        channel: &DynamicImageChannel,
    ) -> Option<&ImageChannel<DynamicSize<Self>>> {
        if let DynamicImageChannel::F32(channel) = channel {
            Some(channel)
        } else {
            None
        }
    }

    fn try_from_dynamic_image_mut(
        channel: &mut DynamicImageChannel,
    ) -> Option<&mut ImageChannel<DynamicSize<Self>>> {
        if let DynamicImageChannel::F32(channel) = channel {
            Some(channel)
        } else {
            None
        }
    }
}

pub trait RuntimePixelType: Clone + Sized + 'static {
    type Primitive: PixelTypePrimitive;
    type PixelSize: PixelSize + Default;
}

pub trait PixelType: RuntimePixelType + Clone + Sized + 'static {
    const ELEMENTS: NonZeroU8;
}

impl<T: PixelTypePrimitive> RuntimePixelType for T {
    type Primitive = T;
    type PixelSize = ComptimeSize<1>;
}

impl<T: PixelTypePrimitive, const PIXEL_ELEMENTS: usize> RuntimePixelType for [T; PIXEL_ELEMENTS] {
    type Primitive = T;
    type PixelSize = ComptimeSize<PIXEL_ELEMENTS>;
}

impl<T: PixelTypePrimitive> PixelType for T {
    const ELEMENTS: NonZeroU8 = NonZeroU8::MIN;
}

impl<T: PixelTypePrimitive, const PIXEL_ELEMENTS: usize> PixelType for [T; PIXEL_ELEMENTS] {
    const ELEMENTS: NonZeroU8 = {
        const {
            assert!(
                PIXEL_ELEMENTS <= 255,
                "PIXEL_ELEMENTS must be less than 256"
            );
            #[allow(clippy::cast_possible_truncation)]
            NonZeroU8::new(PIXEL_ELEMENTS as u8).unwrap()
        }
    };
}
