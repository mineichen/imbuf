use std::{
    mem::ManuallyDrop,
    num::{NonZeroU8, NonZeroU32},
    ptr,
    sync::Arc,
};

use crate::{
    DynamicImageChannel, DynamicImageChannelView, Image, ImageChannel, InvalidLenError, PixelType,
    PixelTypePrimitive,
    channel::{ChannelFactory, ImageChannelVTable, bytes_per_primitive},
};

struct ArcFactory;

impl<T> Image<T, 1>
where
    T: PixelType,
{
    // Todo: If shared_vec is extendedto support various owned slices, it could be implemented on all channel sizes too
    pub fn new_arc(
        input: Arc<[T]>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        let channel = ImageChannel::new_arc(input, width, height)?;
        Ok(Self::from_channel(channel))
    }
}

impl<TP: PixelType> ImageChannel<TP> {
    pub fn new_arc(
        input: Arc<[TP]>,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<Self, InvalidLenError> {
        DynamicImageChannel::ensure_valid_len(width, height, input.len())?;

        let flat_input = crate::pixel::flat_arc(input);
        unsafe {
            // Safety: The vtable is built for TP::Primitive with TP::ELEMENTS
            let ch = DynamicImageChannel::new_arc_flat_unchecked(
                flat_input,
                width,
                height,
                TP::ELEMENTS,
            );
            Ok(Self::from_dynamic_unchecked(ch))
        }
    }
}

impl DynamicImageChannel {
    /// # Safety
    /// In contrast to the public constructors, `T` doesn't have to be `Pod`.
    /// This is only ok, if the `DynamicImageChannel` lives inside a `ImageChannel`, where it cannot be extracted.
    /// `DynamicImageChannel` themself may never have !Pod primitives, because upcasting just occurs based on size/alignment
    /// You could easily transform `Image<NonZeroU8>` into a `DynamicImage` and turn it into `Image<u8>` and modify it.
    /// The drop function, which assumes `NonZeroU8` could get Zero values, which is UB.
    /// Furthermore, `width`/`height`/`pixel_elements` are not checked against the input len
    pub unsafe fn new_arc_flat_unchecked<T: PixelTypePrimitive>(
        input: Arc<[T]>,
        width: NonZeroU32,
        height: NonZeroU32,
        pixel_elements: NonZeroU8,
    ) -> Self {
        let len = input.len();
        debug_assert_eq!(
            DynamicImageChannelView::ensure_valid_flat_len(width, height, pixel_elements, len),
            Ok(()),
            "Incompatible Buffer-Size"
        );

        let ptr = Arc::into_raw(input).cast::<u8>();
        // Safety: The Arc keeps the buffer alive until the vtable drops it and the vtable is
        // built for `T`
        unsafe {
            DynamicImageChannel::from_raw_parts(
                ptr,
                width,
                height,
                pixel_elements,
                <ArcFactory as ChannelFactory<T>>::VTABLE,
                ptr::without_provenance_mut(len),
            )
        }
    }
}

/// Borrows the `Arc` of a channel created by `new_arc` without taking ownership
unsafe fn arc_view<T>(image: &DynamicImageChannelView<'_>) -> ManuallyDrop<Arc<[T]>> {
    let ptr = ptr::slice_from_raw_parts(
        DynamicImageChannelView::raw_ptr(image).cast::<T>(),
        DynamicImageChannelView::raw_data(image) as usize,
    );
    ManuallyDrop::new(unsafe { Arc::from_raw(ptr) })
}

impl<T: PixelTypePrimitive> ChannelFactory<T> for ArcFactory {
    const VTABLE: &'static ImageChannelVTable = {
        unsafe extern "C-unwind" fn make_mut<T: Clone + PixelTypePrimitive>(
            image: &mut DynamicImageChannelView<'_>,
        ) {
            let mut arc = unsafe { arc_view::<T>(image) };

            if Arc::get_mut(&mut arc).is_none() {
                let new_data = Arc::<[T]>::from(&arc[..]);
                let (width, height, pixel_elements) =
                    (image.width(), image.height(), image.pixel_elements());
                // The borrowed `Arc` view is dropped without releasing its share, as
                // `ManuallyDrop` doesn't run `Drop`. The assignment below drops the old channel
                // via its vtable, which releases the share instead.
                *image = unsafe {
                    DynamicImageChannel::new_arc_flat_unchecked(
                        new_data,
                        width,
                        height,
                        pixel_elements,
                    )
                };
            }
        }
        unsafe extern "C-unwind" fn clear_arc_channel<T>(image: &mut DynamicImageChannelView<'_>) {
            drop(ManuallyDrop::into_inner(unsafe { arc_view::<T>(image) }));
        }

        unsafe extern "C-unwind" fn clone_arc_channel<'a, T: PixelTypePrimitive>(
            image: &DynamicImageChannelView<'a>,
        ) -> DynamicImageChannelView<'a> {
            unsafe {
                DynamicImageChannel::new_arc_flat_unchecked(
                    Arc::clone(&arc_view::<T>(image)),
                    image.width(),
                    image.height(),
                    image.pixel_elements(),
                )
            }
        }

        &ImageChannelVTable {
            drop: clear_arc_channel::<T>,
            clone: clone_arc_channel::<T>,
            make_mut: make_mut::<T>,
            bytes_per_primitive: bytes_per_primitive::<T>,
        }
    };
}

pub(crate) unsafe extern "C-unwind" fn clone_slice_into_arc_channel<'a, T: PixelTypePrimitive>(
    image: &DynamicImageChannelView<'a>,
) -> DynamicImageChannelView<'a> {
    unsafe {
        DynamicImageChannel::new_arc_flat_unchecked(
            // Safety: The vtable is built for T
            Arc::from(image.buffer_flat_unchecked::<T>()),
            image.width(),
            image.height(),
            image.pixel_elements(),
        )
    }
}
