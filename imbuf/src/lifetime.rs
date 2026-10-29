//! Lifetime of the buffers of channels and images

use std::{fmt::Debug, hash::Hash, marker::PhantomData};

mod seal {
    pub trait SealedBufferLifetime {}
}

/// Lifetime of the buffers of a channel or image: [`Owned`] or [`Borrowed`]. See
/// [`ImageBase`](crate::ImageBase), why this is a type parameter instead of a lifetime parameter.
pub trait BufferLifetime:
    Copy + Debug + Eq + Hash + Send + Sync + seal::SealedBufferLifetime
{
}

/// The buffers are owned (or shared with other owners), so they are valid as long as the
/// channel or image exists
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Owned(());

/// The buffers are borrowed for `'a`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Borrowed<'a>(PhantomData<&'a ()>);

impl seal::SealedBufferLifetime for Owned {}
impl BufferLifetime for Owned {}
impl seal::SealedBufferLifetime for Borrowed<'_> {}
impl BufferLifetime for Borrowed<'_> {}
