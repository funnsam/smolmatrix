#![cfg_attr(not(any(test, feature = "std")), no_std)]

use core::fmt;
use core::ops::*;

#[macro_export]
macro_rules! matrix {
    ($w: tt x $h: tt $([$($v: expr),* $(,)?])*) => {{
        let mut m = Matrix::<$w, $h>::new_zeroed();

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
    ($h: tt [$($val: expr),* $(,)?]) => {
        $crate::matrix!(1 x $h $([$val])*)
    };
}

pub type Vector<const H: usize> = Matrix<1, H>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Matrix<const W: usize, const H: usize> {
    pub inner: [[f32; W]; H],
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
        let f = &f;
        self.inner.iter_mut().for_each(|i| i.iter_mut().for_each(f));
        self
    }

    pub fn map_zip_ref<F: Fn((&mut f32, &f32))>(mut self, r: &Self, f: F) -> Self {
        let f = &f;
        self.inner.iter_mut().zip(r.inner.iter()).for_each(|(i, j)| i.iter_mut().zip(j.iter()).for_each(f));
        self
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

impl<const W: usize, const H: usize> Add<&Self> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn add(self, b: &Matrix<W, H>) -> Matrix<W, H> {
        self.map_zip_ref(b, |(i, j)| *i += j)
    }
}

impl<const W: usize, const H: usize> Add<f32> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn add(mut self, b: f32) -> Matrix<W, H> {
        for i in self.inner.iter_mut() {
            for j in i.iter_mut() {
                *j += b;
            }
        }

        self
    }
}

impl<const W: usize, const H: usize> Sub<&Self> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn sub(self, b: &Matrix<W, H>) -> Matrix<W, H> {
        self.map_zip_ref(b, |(i, j)| *i -= j)
    }
}

impl<const W: usize, const H: usize> Sub<f32> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn sub(mut self, b: f32) -> Matrix<W, H> {
        for i in self.inner.iter_mut() {
            for j in i.iter_mut() {
                *j -= b;
            }
        }

        self
    }
}

impl<const WAHB: usize, const HA: usize, const WB: usize> Mul<&Matrix<WB, WAHB>>
    for &Matrix<WAHB, HA>
{
    type Output = Matrix<WB, HA>;

    fn mul(self, b: &Matrix<WB, WAHB>) -> Matrix<WB, HA> {
        let mut c = Matrix::new_zeroed();

        for y in 0..HA {
            for x in 0..WB {
                let mut dot = 0.0;

                for i in 0..WAHB {
                    dot += self[(i, y)] * b[(x, i)];
                }

                c[(x, y)] = dot;
            }
        }

        c
    }
}

impl<const W: usize, const H: usize> Mul<f32> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn mul(self, b: f32) -> Matrix<W, H> {
        self.map_each(|i| *i *= b)
    }
}

impl<const H: usize> Mul<&Self> for Vector<H> {
    type Output = Vector<H>;

    fn mul(self, b: &Self) -> Vector<H> {
        self.map_zip_ref(b, |(i, j)| *i *= j)
    }
}

impl<const W: usize, const H: usize> Div<f32> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn div(self, b: f32) -> Matrix<W, H> {
        self * (1.0 / b)
    }
}

impl<const W: usize, const H: usize> Div<&Self> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn div(self, b: &Matrix<W, H>) -> Matrix<W, H> {
        self.map_zip_ref(b, |(i, j)| *i /= j)
    }
}

impl<const W: usize, const H: usize> Neg for Matrix<W, H> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map_each(|i| *i = -*i)
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
}
