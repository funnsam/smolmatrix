use core::ops::*;
use crate::*;

impl<D: Dimension> Add<&Self> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i + j)
    }
}

impl<D: Dimension> Add<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn add(self, rhs: f32) -> Self {
        self.map_each(|i| i + rhs)
    }
}

impl<D: Dimension> Sub<&Self> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i - j)
    }
}

impl<D: Dimension> Sub<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self {
        self.map_each(|i| i - rhs)
    }
}

impl<D: Dimension> Mul<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        self.map_each(|i| i * rhs)
    }
}

impl<D: Dimension> Div<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        self.map_each(|i| i / rhs)
    }
}

impl<D: Dimension> Div<&Self> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i / j)
    }
}

impl<D: Dimension> Neg for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map_each(|i| -i)
    }
}

impl<D: Dimension> AddAssign<&Self> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn add_assign(&mut self, rhs: &Self) {
        self.map_zip_ref_in_place(rhs, |i, j| i + j)
    }
}

impl<D: Dimension> AddAssign<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn add_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i + rhs)
    }
}

impl<D: Dimension> SubAssign<&Self> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn sub_assign(&mut self, rhs: &Self) {
        self.map_zip_ref_in_place(rhs, |i, j| i - j)
    }
}

impl<D: Dimension> SubAssign<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn sub_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i - rhs)
    }
}

impl<D: Dimension> MulAssign<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn mul_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i * rhs)
    }
}

impl<D: Dimension> DivAssign<f32> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn div_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i / rhs)
    }
}

impl<D: Dimension> DivAssign<&Self> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn div_assign(&mut self, rhs: &Self) {
        self.map_zip_ref_in_place(rhs, |i, j| i / j)
    }
}

impl<const WAHB: usize, const HA: usize, const WB: usize> Mul<&Matrix<WB, WAHB>>
    for &Matrix<WAHB, HA>
    where
        [f32; <Dim2<WAHB, HA> as Dimension>::NUM_ELEMENTS]: Sized,
        [f32; <Dim2<WB, WAHB> as Dimension>::NUM_ELEMENTS]: Sized,
        [f32; <Dim2<WB, HA> as Dimension>::NUM_ELEMENTS]: Sized,
        [usize; <Dim2<WB, HA> as Dimension>::ORDER]: Sized,
{
    type Output = Matrix<WB, HA>;

    fn mul(self, rhs: &Matrix<WB, WAHB>) -> Matrix<WB, HA> {
        let mut c = Matrix::new_filled(0.0);

        for y in 0..HA {
            for x in 0..WB {
                let mut dot = 0.0;

                for i in 0..WAHB {
                    dot += self[[i, y]] * rhs[[x, i]];
                }

                c[[x, y]] = dot;
            }
        }

        c
    }
}

impl<D: Dimension> Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    /// Computes the Hadamard product (aka. element-wise multiplication)
    pub fn hadamard_product(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i * j)
    }
}

impl<const W: usize, const H: usize> Matrix<W, H> where [f32; <Dim2<W, H> as Dimension>::NUM_ELEMENTS]: Sized {
    pub fn transpose(self) -> Matrix<H, W> where [f32; <Dim2<H, W> as Dimension>::NUM_ELEMENTS]: Sized {
        let mut inner = [0.0; <Dim2<H, W> as Dimension>::NUM_ELEMENTS];

        for y in 0..H {
            for x in 0..W {
                inner[y + x * H] = self.inner[x + y * W];
            }
        }

        Matrix { inner }
    }
}

impl<const W: usize> HVector<W> where [f32; <Dim1<W> as Dimension>::NUM_ELEMENTS]: Sized {
    #[cfg(feature = "std")]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        let mut acc = 0.0;

        for x in 0..W {
            acc += self[x] * self[x];
        }

        acc
    }

    #[cfg(feature = "std")]
    pub fn unit(self) -> Self {
        let len = self.length();
        self / len
    }

    pub fn dot(&self, b: &Self) -> f32 {
        let mut dot = 0.0;

        for x in 0..W {
            dot += self[x] * b[x];
        }

        dot
    }
}

impl<const H: usize> Vector<H> where [f32; <Dim2<1, H> as Dimension>::NUM_ELEMENTS]: Sized {
    #[cfg(feature = "std")]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        let mut acc = 0.0;

        for y in 0..H {
            acc += self[y] * self[y];
        }

        acc
    }

    #[cfg(feature = "std")]
    pub fn unit(self) -> Self {
        let len = self.length();
        self / len
    }

    pub fn dot(&self, b: &Self) -> f32 {
        let mut dot = 0.0;

        for y in 0..H {
            dot += self[y] * b[y];
        }

        dot
    }
}

impl Vector<3> {
    pub fn cross(&self, b: &Self) -> Self {
        vector!(3 [
            self[1] * b[2] - self[2] * b[1],
            self[2] * b[0] - self[0] * b[2],
            self[0] * b[1] - self[1] * b[0],
        ])
    }
}

impl HVector<3> {
    pub fn cross(&self, b: &Self) -> Self {
        hvector!(3 [
            self[1] * b[2] - self[2] * b[1],
            self[2] * b[0] - self[0] * b[2],
            self[0] * b[1] - self[1] * b[0],
        ])
    }
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
#[test]
fn vector3_cross() {
    let a = vector!(3 [5.0, 6.0, 2.0]);
    let b = vector!(3 [1.0, 1.0, 1.0]);

    assert_eq!(a.cross(&b), vector!(3 [4.0, -3.0, -1.0]));
}

#[cfg(test)]
#[test]
fn add_assi() {
    let mut a = vector!(3 [5.0, 6.0, 2.0]);
    let b = vector!(3 [1.0, 1.0, 1.0]);

    a += &b;
    assert_eq!(a, vector!(3 [6.0, 7.0, 3.0]));

    a += 1.0;
    assert_eq!(a, vector!(3 [7.0, 8.0, 4.0]));
}
