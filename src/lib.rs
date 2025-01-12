#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use core::fmt;

mod index;
mod ops;

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tensor<D: Dimension> where [f32; D::NUM_ELEMENTS]: Sized {
    pub inner: [f32; D::NUM_ELEMENTS],
}

pub trait Dimension {
    const ORDER: usize;
    const DIMENSIONS: &[usize];
    const NUM_ELEMENTS: usize;
}

impl<D: Dimension> Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
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
            [f32; $fn::<$($ti,)* 1>::NUM_ELEMENTS]: Sized,
            [f32; $tn::<$($ti),*>::NUM_ELEMENTS]: Sized,
        {
            fn from(value: Tensor<$fn<$($ti,)* 1>>) -> Self {
                // SAFETY: A tensor of size […, 1] and […] is the same
                unsafe {
                    Self { inner: *core::mem::transmute::<&[f32; <$fn<$($ti,)* 1> as Dimension>::NUM_ELEMENTS], &[f32; <$tn<$($ti),*> as Dimension>::NUM_ELEMENTS]>(&value.inner) }
                }
            }
        }

        impl<$(const $ti: usize),*> From<Tensor<$tn<$($ti),*>>> for Tensor<$fn<$($ti,)* 1>> where
            [f32; $fn::<$($ti,)* 1>::NUM_ELEMENTS]: Sized,
            [f32; $tn::<$($ti),*>::NUM_ELEMENTS]: Sized,
        {
            fn from(value: Tensor<$tn<$($ti),*>>) -> Self {
                // SAFETY: A tensor of size […] and […, 1] is the same
                unsafe {
                    Self { inner: *core::mem::transmute::<&[f32; <$tn<$($ti),*> as Dimension>::NUM_ELEMENTS], &[f32; <$fn<$($ti,)* 1> as Dimension>::NUM_ELEMENTS]>(&value.inner) }
                }
            }
        }
    };
    ($n:tt $t:tt $d:tt $($i:tt),*) => {
        /// Marker object representing a $d-dimensional size. This struct contains a private empty
        /// tuple so that it isn't constructable.
        #[derive(Debug, Clone, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $n<$(const $i: usize),*>(pub(self) ());

        impl<$(const $i: usize),*> Dimension for $n<$($i),*> {
            const ORDER: usize = $d;
            const DIMENSIONS: &[usize] = &[$($i),*];
            const NUM_ELEMENTS: usize = 1 $(* $i)*;
        }

        /// A $d-th order tensor.
        pub type $t<$(const $i: usize),*> = Tensor<$n<$($i),*>>;
    };
}

pub type Vector<const H: usize> = Matrix<1, H>;

dim!(Dim0 Scalar 0);
dim!(Dim1 HVector 1 W);
dim!(Dim2 Matrix 2 W, H);
dim!(Dim3 Tensor3 3 W, H, D);
dim!(Dim4 Tensor4 4 X, Y, Z, W);
dim!(Dim5 Tensor5 5 X, Y, Z, W, V);
dim!(Dim6 Tensor6 6 X, Y, Z, W, V, U);

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
    ($w: tt x $h: tt $([$($v: expr),* $(,)?])*) => {{
        let mut m = $crate::Matrix::<$w, $h>::new_filled(0.0);

        $crate::matrix_fill!(m, 0, $([$($v,)*])*);

        m
    }};
}

#[macro_export]
macro_rules! matrix_fill {
    ($m: expr, $y: expr, [$($v: expr,)+] $($rest: tt)*) => {{
        $crate::matrix_fill!($m, $y, 0, $($v,)*);
        $crate::matrix_fill!($m, $y + 1, $($rest)*);
    }};
    ($m: expr, $y: expr, $x: expr, $v: expr, $($rest: tt)*) => {{
        $m[[$x, $y]] = $v;
        $crate::matrix_fill!($m, $y, $x + 1, $($rest)*);
    }};

    ($m: expr, $y: expr,) => {};
    ($m: expr, $y: expr, $x: expr,) => {};
}

#[macro_export]
macro_rules! vector_swap {
    (@ $at: expr, $vector: expr, $orig: expr, $idx: tt $($rest: tt)*) => {{
        $vector[$at] = $orig[$idx];
        crate::vector_swap!(@ $at + 1, $vector, $orig, $($rest)*);
    }};
    (@ $at: expr, $vector: expr, $orig: expr,) => {};
    ($vector: expr, $($idx: tt)*) => {{
        let mut og = $vector;
        let mut v = $vector.clone();
        crate::vector_swap!(@ 0, v, og, $($idx)*);
        v
    }};
}

impl<D: Dimension> FromIterator<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        Self { inner: core::array::from_fn(|_| iter.next().unwrap()) }
    }
}

impl<const W: usize, const H: usize> fmt::Display for Matrix<W, H> where
    [f32; <Dim2<W, H> as Dimension>::NUM_ELEMENTS]: Sized,
    [usize; <Dim2<W, H> as Dimension>::ORDER]: Sized,
{
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
    let a: Scalar = hvector!(1 [1.2]).into();
    assert_eq!(a, Scalar { inner: [1.2] });

    let a: HVector<2> = matrix!(2 x 1 [1.2, 2.3]).into();
    assert_eq!(a, HVector { inner: [1.2, 2.3] });
}

#[cfg(test)]
#[test]
fn up_conv() {
    let a: HVector<1> = Scalar::new_filled(1.2).into();
    assert_eq!(a, HVector { inner: [1.2] });

    let a: Matrix<2, 1> = hvector!(2 [1.2, 2.3]).into();
    assert_eq!(a, Matrix { inner: [1.2, 2.3] });

    let a: Tensor3<2, 2, 1> = matrix!(2 x 2 [1.0, 2.0] [3.0, 4.0]).into();
    assert_eq!(a, Tensor3 { inner: [1.0, 2.0, 3.0, 4.0] });
}
