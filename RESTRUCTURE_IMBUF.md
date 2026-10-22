Having ImageChannel<DynamicSize<_>> causes bad ergonomics... Drop this intermediate. Instead
  Make DynamicImageChannel a struct which contains a UnsafeImageChannel and implement `TryInto<ImageChannel<P, N>>`. Remove the Generics from `UnsafeImageChannel` and `DynamicImageChannel` and store a *const u8 instead. DynamicImageChannel contains a UnsafeImageChannel on which `TryInto` can match... Functions inside the VTable remain Generic over the buffer primitive... VTable gets a new member `fn primitive_size() -> NonZeroUsize`... This allows removal of RuntimePixelType. DynamicImageChannel::buffer_flat will require a Type parameter(PixelPrimitive) which implements bytemuck::Pod (today only requires `Clone` beside others). ImageChannel (which is typed), can call buffer_flat on any Type... Therefore, the TryInto from DynamicImageChannel to ImageChannel<T> only works for T: Pod. Unsafe conversions (for e.g. NonZero<u8>) will be added in the future.  Don't extend PixelTypePrimitive with Pod, as I want to be able to still use e.g. `NonZero<u8>` as PixelTypePrimitive for Images and ImageChannel (it will just involve unsafe from the Caller to e.g. create a slice out of a DynamicImageChannel or Map DynamicImageChannel->ImageChannel)... Change all CABI function pointers in vtable to "C unwind", to propagate panic instead of abort in case of a panic... ABI

DynamicImageChannel expects the following invariants over UnsafeImageChannel
- byte_len (Width * Height * VTable::primitive_size() * pixel_elements) < usize::MAX
- ptr must have byte_len elements allocated (which can be shared between instances)
- vtable is able to work with ptr

Update imbuf/changelog.md for a version "0.4"
