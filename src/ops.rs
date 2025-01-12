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
