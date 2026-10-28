use std::num::NonZeroU32;

const NZ_2: NonZeroU32 = NonZeroU32::new(2).unwrap();

use imbuf::Image;
use testresult::TestResult;

#[test]
fn flatten() -> TestResult {
    let input = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]];
    let image = Image::<[u8; 3], 1>::new_vec(input.clone(), NZ_2, NZ_2)?;
    let buffers = image.buffer();
    assert_eq!(buffers, &input);
    assert_eq!(image.buffer_flat(), (0..12).collect::<Vec<_>>());
    Ok(())
}
