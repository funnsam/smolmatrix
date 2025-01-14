#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use core::fmt;

mod index;
mod ops;
pub mod vector;

trait Seal {}

#[macro_export]
/// Used for generic `where` [`Dimension`] bounds.
///
/// # Usage
/// An optional `inner` parameter is passed when you don't need access to indexing.
///
/// # Example
/// ```
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use smolmatrix::*;
///
/// struct MyTensoredStruct<D: Dimension> (Tensor<D>) where bound!(inner D): Sized;
///
/// impl<D: Dimension> MyTensoredStruct<D> where bound!(D): Sized {
///     pub fn get(&self, index: [usize; D::ORDER]) -> f32 {
///         self.0[index]
///     }
/// }
/// ```
macro_rules! bound {
    (inner $dim:ty) => {
        [f32; <$dim as $crate::Dimension>::NUM_ELEMENTS]
    };
    ($dim:ty) => {
        ($crate::bound!(inner $dim), [usize; <$dim as $crate::Dimension>::ORDER])
    };
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tensor<D: Dimension> where bound!(inner D): Sized {
    pub inner: [f32; D::NUM_ELEMENTS],
}

pub trait Dimension {
    const ORDER: usize;
    const DIMENSIONS: &[usize];
    const NUM_ELEMENTS: usize;
}

impl<D: Dimension> Tensor<D> where bound!(inner D): Sized {
    pub fn new_filled(value: f32) -> Self {
        Self { inner: [value; D::NUM_ELEMENTS] }
    }

    pub fn map_each<F: Fn(f32) -> f32>(mut self, f: F) -> Self {
        self.map_each_in_place(f);
        self
    }

    pub fn map_zip_ref<F: Fn(f32, f32) -> f32>(mut self, r: &Self, f: F) -> Self {
        self.map_zip_ref_in_place(r, f);
        self
    }

    pub fn map_each_in_place<F: Fn(f32) -> f32>(&mut self, f: F) {
        let f = &f;
        self.inner.iter_mut().for_each(|i| *i = f(*i));
    }

    pub fn map_zip_ref_in_place<F: Fn(f32, f32) -> f32>(&mut self, r: &Self, f: F) {
        let f = &f;
        self.inner.iter_mut().zip(r.inner.iter()).for_each(|(i, j)| *i = f(*i, *j));
    }
}

macro_rules! dim {
    (conv $fn:tt <=> $tn:tt $($ti:tt),*) => {
        impl<$(const $ti: usize),*> From<Tensor<$fn<$($ti,)* 1>>> for Tensor<$tn<$($ti),*>> where
            bound!(inner $fn::<$($ti,)* 1>): Sized,
            bound!(inner $tn::<$($ti),*>): Sized,
        {
            fn from(value: Tensor<$fn<$($ti,)* 1>>) -> Self {
                value.down_conv()
            }
        }

        impl<$(const $ti: usize),*> From<Tensor<$tn<$($ti),*>>> for Tensor<$fn<$($ti,)* 1>> where
            bound!(inner $fn::<$($ti,)* 1>): Sized,
            bound!(inner $tn::<$($ti),*>): Sized,
        {
            fn from(value: Tensor<$tn<$($ti),*>>) -> Self {
                value.up_conv()
            }
        }

        impl<$(const $ti: usize),*> Tensor<$tn<$($ti),*>> where
            bound!(inner $fn::<$($ti,)* 1>): Sized,
            bound!(inner $tn::<$($ti),*>): Sized,
        {
            pub fn up_conv(self) -> Tensor<$fn<$($ti,)* 1>> {
                // SAFETY: A tensor of size […] and […, 1] is the same
                unsafe {
                    Tensor { inner: *core::mem::transmute::<&[f32; <$tn<$($ti),*> as Dimension>::NUM_ELEMENTS], &[f32; <$fn<$($ti,)* 1> as Dimension>::NUM_ELEMENTS]>(&self.inner) }
                }
            }
        }

        impl<$(const $ti: usize),*> Tensor<$fn<$($ti,)* 1>> where
            bound!(inner $fn::<$($ti,)* 1>): Sized,
            bound!(inner $tn::<$($ti),*>): Sized,
        {
            pub fn down_conv(self) -> Tensor<$tn<$($ti),*>> {
                // SAFETY: A tensor of size […, 1] and […] is the same
                unsafe {
                    Tensor { inner: *core::mem::transmute::<&[f32; <$fn<$($ti,)* 1> as Dimension>::NUM_ELEMENTS], &[f32; <$tn<$($ti),*> as Dimension>::NUM_ELEMENTS]>(&self.inner) }
                }
            }
        }
    };
    ($n:tt $t:tt $d:tt $ord:tt $($i:tt),*) => {
        #[doc = concat!("Marker object representing a ", $d, "-dimensional size. This struct")]
        #[doc = "contains a private empty tuple so that it isn't constructable."]
        #[derive(Debug, Clone, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $n<$(const $i: usize),*>(pub(self) ());

        impl<$(const $i: usize),*> Seal for $n<$($i),*> {}

        impl<$(const $i: usize),*> Dimension for $n<$($i),*> {
            const ORDER: usize = $d;
            const DIMENSIONS: &[usize] = &[$($i),*];
            const NUM_ELEMENTS: usize = 1 $(* $i)*;
        }

        #[doc = concat!("A ", $d, "-", $ord, " order tensor.")]
        pub type $t<$(const $i: usize),*> = Tensor<$n<$($i),*>>;
    };
}

/// Type alias of a vertical vector.
pub type Vector<const H: usize> = Matrix<1, H>;

dim!(Dim0 Scalar  0 "th");
dim!(Dim1 HVector 1 "st" W);
dim!(Dim2 Matrix  2 "nd" W, H);
dim!(Dim3 Tensor3 3 "rd" W, H, D);
dim!(Dim4 Tensor4 4 "th" X, Y, Z, W);
dim!(Dim5 Tensor5 5 "th" X, Y, Z, W, V);
dim!(Dim6 Tensor6 6 "th" X, Y, Z, W, V, U);

dim!(conv Dim1 <=> Dim0);
dim!(conv Dim2 <=> Dim1 W);
dim!(conv Dim3 <=> Dim2 W, H);
dim!(conv Dim4 <=> Dim3 W, H, D);
dim!(conv Dim5 <=> Dim4 X, Y, Z, W);
dim!(conv Dim6 <=> Dim5 X, Y, Z, W, V);

#[macro_export]
macro_rules! vector {
    ($size:tt $e:expr) => { $crate::Vector::<$size> { inner: $e } };
}

#[macro_export]
macro_rules! hvector {
    ($size:tt $e:expr) => { $crate::HVector::<$size> { inner: $e } };
}

#[macro_export]
macro_rules! matrix {
    ($w:tt x $h:tt $([$($v:expr),* $(,)?])*) => {{
        let mut m = $crate::Matrix::<$w, $h>::new_filled(0.0);

        $crate::matrix_fill!(m, 0, $([$($v,)*])*);

        m
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! matrix_fill {
    ($m:expr, $y:expr, [$($v:expr,)+] $($rest:tt)*) => {{
        $crate::matrix_fill!($m, $y, 0, $($v,)*);
        $crate::matrix_fill!($m, $y + 1, $($rest)*);
    }};
    ($m: expr, $y: expr, $x: expr, $v: expr, $($rest:tt)*) => {{
        $m[[$x, $y]] = $v;
        $crate::matrix_fill!($m, $y, $x + 1, $($rest)*);
    }};

    ($m: expr, $y: expr,) => {};
    ($m: expr, $y: expr, $x: expr,) => {};
}

#[doc(hidden)]
#[macro_export]
macro_rules! _vector_swap {
    ($at:expr, $vector:expr, $orig:expr, $idx:tt $($rest:tt)*) => {{
        $vector[$at] = $orig[$idx];
        $crate::_vector_swap!($at + 1, $vector, $orig, $($rest)*);
    }};
    ($at:expr, $vector:expr, $orig:expr,) => {};
}

#[macro_export]
macro_rules! vector_swap {
    ($vector:expr, $($idx:tt)*) => {{
        let mut og = $vector;
        let mut v = $vector.clone();
        $crate::_vector_swap!(0, v, og, $($idx)*);
        v
    }};
}

impl<D: Dimension> FromIterator<f32> for Tensor<D> where bound!(inner D): Sized {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        Self { inner: core::array::from_fn(|_| iter.next().unwrap()) }
    }
}

impl<const W: usize, const H: usize> fmt::Display for Matrix<W, H> where bound!(Dim2<W, H>): Sized {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        struct LenOf {
            len: usize,
        }

        impl fmt::Write for LenOf {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                self.len += s.len();
                Ok(())
            }
        }

        let mut align = [0; W];
        for y in 0..H {
            for x in 0..W {
                use core::fmt::Write;

                let mut len = LenOf { len: 0 };
                write!(&mut len, "{}", self.inner[Self::index_of([x, y])])?;
                align[x] = align[x].max(len.len);
            }
        }

        let inner_space = align.iter().sum::<usize>() + W + 1;
        writeln!(f, "┌{:<inner_space$}┐", "")?;

        for y in 0..H {
            write!(f, "│ ")?;

            for x in 0..W {
                write!(f, "{:^1$} ", self.inner[Self::index_of([x, y])], align[x])?;
            }

            writeln!(f, "│")?;
        }

        writeln!(f, "└{:<inner_space$}┘", "")
    }
}

#[cfg(test)]
#[test]
fn down_conv() {
    let a = hvector!(1 [1.2]).down_conv();
    assert_eq!(a, Scalar { inner: [1.2] });

    let a = matrix!(2 x 1 [1.2, 2.3]).down_conv();
    assert_eq!(a, HVector { inner: [1.2, 2.3] });
}

#[cfg(test)]
#[test]
fn up_conv() {
    let a = Scalar::new_filled(1.2).up_conv();
    assert_eq!(a, HVector { inner: [1.2] });

    let a = hvector!(2 [1.2, 2.3]).up_conv();
    assert_eq!(a, Matrix { inner: [1.2, 2.3] });

    let a = matrix!(2 x 2 [1.0, 2.0] [3.0, 4.0]).up_conv();
    assert_eq!(a, Tensor3 { inner: [1.0, 2.0, 3.0, 4.0] });
}
