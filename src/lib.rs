#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

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
    (conv_down $fn:tt => $tn:tt $($ti:tt),*) => {
        impl<$(const $ti: usize),*> From<Tensor<$fn<$($ti,)* 1>>> for Tensor<$tn<$($ti),*>> where
            [f32; $fn::<$($ti,)* 1>::NUM_ELEMENTS]: Sized,
            [f32; $tn::<$($ti),*>::NUM_ELEMENTS]: Sized,
        {
            fn from(value: Tensor<$fn<$($ti,)* 1>>) -> Self {
                // SAFETY: A tensor of size [.., 1] and [..] is the same
                unsafe {
                    Self { inner: *core::mem::transmute::<&[f32; <$fn::<$($ti,)* 1> as Dimension>::NUM_ELEMENTS], &[f32; <$tn::<$($ti),*> as Dimension>::NUM_ELEMENTS]>(&value.inner) }
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

dim!(conv_down Dim1 => Dim0);
dim!(conv_down Dim2 => Dim1 W);
dim!(conv_down Dim3 => Dim2 W, H);

/*
#[macro_export]
macro_rules! matrix {
    ($w: tt x $h: tt $([$($v: expr),* $(,)?])*) => {{
        let mut m = $crate::Matrix::<$w, $h>::new_zeroed();

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
        $m[($x, $y)] = $v;
        $crate::matrix_fill!($m, $y, $x + 1, $($rest)*);
    }};

    ($m: expr, $y: expr,) => {};
    ($m: expr, $y: expr, $x: expr,) => {};
}

#[macro_export]
macro_rules! vector {
    ($h: tt [$($val: expr),* $(,)?]) => {{
        $crate::matrix!(1 x $h $([$val])*)
    }};
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

impl<const W: usize, const H: usize> FromIterator<f32> for Matrix<W, H> {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let mut m = Self::new_zeroed();

        for y in 0..H {
            for x in 0..W {
                m[(x, y)] = iter.next().unwrap();
            }
        }

        m
    }
}

impl<const W: usize, const H: usize> Matrix<W, H> {
    pub const fn new_zeroed() -> Self {
        Self {
            inner: [[0.0; W]; H],
        }
    }

    pub fn transpose(&self) -> Matrix<H, W> {
        let mut iters = self.inner.map(|r| r.into_iter());
        let m = core::array::from_fn(|_| core::array::from_fn(|i| iters[i].next().unwrap()));
        Matrix { inner: m }
    }

    pub fn map_each<F: Fn(&mut f32)>(mut self, f: F) -> Self {
        self.map_each_in_place(f);
        self
    }

    pub fn map_zip_ref<F: Fn((&mut f32, &f32))>(mut self, r: &Self, f: F) -> Self {
        self.map_zip_ref_in_place(r, f);
        self
    }

    pub fn map_each_in_place<F: Fn(&mut f32)>(&mut self, f: F) {
        let f = &f;
        self.inner.iter_mut().for_each(|i| i.iter_mut().for_each(f));
    }

    pub fn map_zip_ref_in_place<F: Fn((&mut f32, &f32))>(&mut self, r: &Self, f: F) {
        let f = &f;
        self.inner.iter_mut().zip(r.inner.iter()).for_each(|(i, j)| i.iter_mut().zip(j.iter()).for_each(f));
    }
}

impl<const H: usize> Vector<H> {
    #[cfg(feature = "std")]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        let mut acc = 0.0;

        for y in 0..H {
            acc += self[(0, y)] * self[(0, y)];
        }

        acc
    }

    #[cfg(feature = "std")]
    #[deprecated]
    pub fn magnitude(self) -> Self {
        self.unit()
    }

    #[cfg(feature = "std")]
    pub fn unit(self) -> Self {
        let len = self.length();
        self / len
    }

    pub fn dot(&self, b: &Self) -> f32 {
        let mut dot = 0.0;

        for y in 0..H {
            dot += self[(0, y)] * b[(0, y)];
        }

        dot
    }

    pub fn x(&self) -> f32 { self[0] }
    pub fn y(&self) -> f32 { self[1] }
    pub fn z(&self) -> f32 { self[2] }
    pub fn w(&self) -> f32 { self[3] }

    pub fn x_ref(&self) -> &f32 { &self[0] }
    pub fn y_ref(&self) -> &f32 { &self[1] }
    pub fn z_ref(&self) -> &f32 { &self[2] }
    pub fn w_ref(&self) -> &f32 { &self[3] }

    pub fn x_mut(&mut self) -> &mut f32 { &mut self[0] }
    pub fn y_mut(&mut self) -> &mut f32 { &mut self[1] }
    pub fn z_mut(&mut self) -> &mut f32 { &mut self[2] }
    pub fn w_mut(&mut self) -> &mut f32 { &mut self[3] }
}

impl Vector<3> {
    pub fn cross(&self, b: &Self) -> Self {
        vector!(
            3 [
                self[1] * b[2] - self[2] * b[1],
                self[2] * b[0] - self[0] * b[2],
                self[0] * b[1] - self[1] * b[0],
            ]
        )
    }
}

impl<const W: usize, const H: usize> Index<(usize, usize)> for Matrix<W, H> {
    type Output = f32;

    fn index(&self, i: (usize, usize)) -> &f32 {
        &self.inner[i.1][i.0]
    }
}

impl<const W: usize, const H: usize> IndexMut<(usize, usize)> for Matrix<W, H> {
    fn index_mut(&mut self, i: (usize, usize)) -> &mut f32 {
        &mut self.inner[i.1][i.0]
    }
}

impl<const H: usize> Index<usize> for Vector<H> {
    type Output = f32;

    fn index(&self, i: usize) -> &f32 {
        &self.inner[i][0]
    }
}

impl<const H: usize> IndexMut<usize> for Vector<H> {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.inner[i][0]
    }
}

impl<const W: usize, const H: usize> fmt::Display for Matrix<W, H> {
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
        for y in self.inner.iter() {
            for (xi, x) in y.iter().enumerate() {
                use core::fmt::Write;

                let mut len = LenOf { len: 0 };
                write!(&mut len, "{x}")?;
                align[xi] = align[xi].max(len.len);
            }
        }

        let inner_space = align.iter().sum::<usize>() + W + 1;
        writeln!(f, "┌{:<inner_space$}┐", "")?;

        for y in self.inner.iter() {
            write!(f, "│ ")?;

            for (xi, x) in y.iter().enumerate() {
                write!(f, "{x:^0$} ", align[xi])?;
            }

            writeln!(f, "│")?;
        }

        writeln!(f, "└{:<inner_space$}┘", "")
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_mul() {
        let a = matrix!(
            3 x 2
            [1.0, 2.0, 3.0]
            [4.0, 5.0, 6.0]
        );
        let b = matrix!(
            2 x 3
            [10.0, 11.0]
            [20.0, 21.0]
            [30.0, 31.0]
        );

        let c = &a * &b;

        println!("{a}"); // cargo test -- --nocapture
        println!("{b}");
        println!("{c}");

        assert_eq!(c, matrix!(
            2 x 2
            [140.0, 146.0]
            [320.0, 335.0]
        ));

        let d = c * 2.0;
        assert_eq!(d, matrix!(
            2 x 2
            [280.0, 292.0]
            [640.0, 670.0]
        ));
    }

    #[test]
    fn matrix_add() {
        let a = matrix!(
            3 x 2
            [1.0, -1.0, 2.0]
            [0.0, 3.0, 4.0]
        );
        let b = matrix!(
            3 x 2
            [2.0, -1.0, 5.0]
            [7.0, 1.0, 4.0]
        );

        println!("{a}"); // cargo test -- --nocapture

        let c = a + &b;

        println!("{b}");
        println!("{c}");

        assert_eq!(c, matrix!(
            3 x 2
            [3.0, -2.0, 7.0]
            [7.0, 4.0, 8.0]
        ));
    }

    #[test]
    fn matrix_transpose() {
        let m = matrix!(
            3 x 2
            [1.0, 2.0, 3.0]
            [4.0, 5.0, 6.0]
        );

        assert_eq!(m.transpose(), matrix!(
            2 x 3
            [1.0, 4.0]
            [2.0, 5.0]
            [3.0, 6.0]
        ));
    }

    #[test]
    fn vector3_cross() {
        let a = vector!(3 [5.0, 6.0, 2.0]);
        let b = vector!(3 [1.0, 1.0, 1.0]);

        assert_eq!(a.cross(&b), vector!(3 [4.0, -3.0, -1.0]));
    }

    #[test]
    fn add_assi() {
        let mut a = vector!(3 [5.0, 6.0, 2.0]);
        let b = vector!(3 [1.0, 1.0, 1.0]);

        a += &b;
        assert_eq!(a, vector!(3 [6.0, 7.0, 3.0]));

        a += 1.0;
        assert_eq!(a, vector!(3 [7.0, 8.0, 4.0]));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        use serde_test::*;

        let a = matrix!(
            3 x 2
            [1.0, 3.0, 5.0]
            [2.0, 4.0, 6.0]
        );
        assert_tokens(
            &a,
            &[
                Token::Tuple { len: 6 },
                Token::F32(1.0),
                Token::F32(3.0),
                Token::F32(5.0),
                Token::F32(2.0),
                Token::F32(4.0),
                Token::F32(6.0),
                Token::TupleEnd,
            ],
        );
    }
}
*/
