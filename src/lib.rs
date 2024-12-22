#![allow(non_snake_case, clippy::too_many_arguments)]

#[macro_export]
macro_rules! concat {
    () => {
        ::core::compile_error!("signtensors::concat! requires at least one argument")
    };
    ($($sct: expr),+ $(,)?) => {
        $crate::Concat::concat(&[$(($sct).as_ref(),)+])
    };
}
use equator::assert;
use faer::SimpleEntity;
use reborrow::*;

#[doc(hidden)]
pub mod bitmagic;

pub mod inplace_sct;
pub mod sct;

pub mod sct_tensor;

trait Storage: SimpleEntity {}
impl Storage for u8 {}
impl Storage for u16 {}
impl Storage for u32 {}
impl Storage for u64 {}

pub struct SignMatRef<'a> {
    storage: MatRef<'a, u64>,
    nrows: usize,
}

pub struct SignMatMut<'a> {
    storage: MatMut<'a, u64>,
    nrows: usize,
}

pub use faer::{MatMut, MatRef};

impl<'a> SignMatRef<'a> {
    #[inline]
    #[track_caller]
    pub fn from_storage(storage: MatRef<'a, u64>, nrows: usize) -> Self {
        assert!(storage.nrows() * 64 >= nrows);
        Self { storage, nrows }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.storage.ncols()
    }

    #[inline]
    pub fn storage(self) -> MatRef<'a, u64> {
        self.storage
    }

    #[inline]
    fn storage_as<T: Storage>(self) -> MatRef<'a, T> {
        unsafe {
            faer::mat::from_raw_parts(
                self.storage.as_ptr() as *const T,
                self.nrows().div_ceil(core::mem::size_of::<T>()),
                self.ncols(),
                1,
                self.storage.col_stride()
                    * (core::mem::size_of::<u64>() / core::mem::size_of::<T>()) as isize,
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col(self, col: usize) -> (SignMatRef<'a>, SignMatRef<'a>) {
        let nrows = self.nrows();
        let (left, right) = self.storage().split_at_col(col);
        (
            Self::from_storage(left, nrows),
            Self::from_storage(right, nrows),
        )
    }
}

impl<'a> SignMatMut<'a> {
    #[inline]
    #[track_caller]
    pub fn from_storage(storage: MatMut<'a, u64>, nrows: usize) -> Self {
        assert!(storage.nrows() * 64 >= nrows);
        Self { storage, nrows }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.storage.ncols()
    }

    #[inline]
    pub fn storage(self) -> MatRef<'a, u64> {
        self.storage.into_const()
    }

    #[inline]
    pub fn storage_mut(self) -> MatMut<'a, u64> {
        self.storage
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(self, col: usize) -> (SignMatMut<'a>, SignMatMut<'a>) {
        let nrows = self.nrows();
        let (left, right) = self.storage_mut().split_at_col_mut(col);
        (
            Self::from_storage(left, nrows),
            Self::from_storage(right, nrows),
        )
    }
}

impl Copy for SignMatRef<'_> {}
impl Clone for SignMatRef<'_> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'short> Reborrow<'short> for SignMatRef<'_> {
    type Target = SignMatRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short> ReborrowMut<'short> for SignMatRef<'_> {
    type Target = SignMatRef<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a> IntoConst for SignMatRef<'a> {
    type Target = SignMatRef<'a>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'short> Reborrow<'short> for SignMatMut<'_> {
    type Target = SignMatRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        SignMatRef {
            storage: self.storage.rb(),
            nrows: self.nrows,
        }
    }
}
impl<'short> ReborrowMut<'short> for SignMatMut<'_> {
    type Target = SignMatMut<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        SignMatMut {
            storage: self.storage.rb_mut(),
            nrows: self.nrows,
        }
    }
}
impl<'a> IntoConst for SignMatMut<'a> {
    type Target = SignMatRef<'a>;

    #[inline]
    fn into_const(self) -> Self::Target {
        SignMatRef {
            storage: self.storage.into_const(),
            nrows: self.nrows,
        }
    }
}

pub trait Concat: Copy {
    type Owned;
    fn concat(list: &[Self]) -> Self::Owned;
}
