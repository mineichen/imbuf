#![doc = include_str!("../README.md")]

mod from_imbuf;
mod into_imbuf;

pub use from_imbuf::{ImbufAsOpencv, ImbufAsOpencvMut};
pub use into_imbuf::OpencvIntoImbuf;
