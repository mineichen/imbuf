//! Owned images have to stay usable in `Send` futures, which hold them in closure based adapters
//! across `.await`. See the documentation of `ImageBase`, why owned images have no lifetime
//! parameter.

use std::future::Future;

use imbuf::{DynamicImage, DynamicImageChannel, Image, ImageChannel};

fn assert_send<F: Future + Send>(_: F) {}

/// `Peekable` stores `<Map<_, F> as Iterator>::Item`, so proving `Send` for the future requires the
/// signature of the closure, like most adapters of async streams (e.g. `buffer_unordered`).
/// The future must not be generic over the image type, otherwise its signature doesn't contain
/// the concrete type.
macro_rules! hold_mapped_across_await {
    ($ty:ty) => {
        async {
            let mut iter = Vec::<$ty>::new().into_iter().map(|x| x).peekable();
            std::future::ready(()).await;
            drop(iter.next());
        }
    };
}

#[test]
fn owned_images_in_closure_held_across_await_are_send() {
    assert_send(hold_mapped_across_await!(DynamicImage));
    assert_send(hold_mapped_across_await!(DynamicImageChannel));
    assert_send(hold_mapped_across_await!(Image<u8, 3>));
    assert_send(hold_mapped_across_await!(ImageChannel<[u16; 3]>));
}
