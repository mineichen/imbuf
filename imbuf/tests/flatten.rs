use std::num::NonZeroU32;

use imbuf::Image;

#[test]
fn flatten() {
    let two = NonZeroU32::new(2).unwrap();
    let input = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]];
    let image = Image::<[u8; 3], 1>::new_vec(input.clone(), two, two);
    let buffers = image.buffer();
    assert_eq!(buffers, &input);
    assert_eq!(image.buffer_flat(), (0..12).collect::<Vec<_>>());
}
