use core::ops::*;
use crate::*;

impl<D: Dimension> Add<&Self> for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i + j)
    }
}

impl<D: Dimension> Add<f32> for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn add(self, rhs: f32) -> Self {
        self.map_each(|i| i + rhs)
    }
}

impl<D: Dimension> Sub<&Self> for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i - j)
    }
}

impl<D: Dimension> Sub<f32> for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self {
        self.map_each(|i| i - rhs)
    }
}

impl<D: Dimension> Mul<f32> for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        self.map_each(|i| i * rhs)
    }
}

impl<D: Dimension> Div<f32> for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        self.map_each(|i| i / rhs)
    }
}

impl<D: Dimension> Div<&Self> for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i / j)
    }
}

impl<D: Dimension> Neg for Tensor<D> where bound!(inner D): Sized {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map_each(|i| -i)
    }
}

impl<D: Dimension> AddAssign<&Self> for Tensor<D> where bound!(inner D): Sized {
    fn add_assign(&mut self, rhs: &Self) {
        self.map_zip_ref_in_place(rhs, |i, j| i + j)
    }
}

impl<D: Dimension> AddAssign<f32> for Tensor<D> where bound!(inner D): Sized {
    fn add_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i + rhs)
    }
}

impl<D: Dimension> SubAssign<&Self> for Tensor<D> where bound!(inner D): Sized {
    fn sub_assign(&mut self, rhs: &Self) {
        self.map_zip_ref_in_place(rhs, |i, j| i - j)
    }
}

impl<D: Dimension> SubAssign<f32> for Tensor<D> where bound!(inner D): Sized {
    fn sub_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i - rhs)
    }
}

impl<D: Dimension> MulAssign<f32> for Tensor<D> where bound!(inner D): Sized {
    fn mul_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i * rhs)
    }
}

impl<D: Dimension> DivAssign<f32> for Tensor<D> where bound!(inner D): Sized {
    fn div_assign(&mut self, rhs: f32) {
        self.map_each_in_place(|i| i / rhs)
    }
}

impl<D: Dimension> DivAssign<&Self> for Tensor<D> where bound!(inner D): Sized {
    fn div_assign(&mut self, rhs: &Self) {
        self.map_zip_ref_in_place(rhs, |i, j| i / j)
    }
}

impl<const WAHB: usize, const HA: usize, const WB: usize> Mul<&Matrix<WB, WAHB>>
    for &Matrix<WAHB, HA>
    where
        bound!(inner Dim2<WAHB, HA>): Sized,
        bound!(inner Dim2<WB, WAHB>): Sized,
        bound!(Dim2<WB, HA>): Sized,
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

impl<D: Dimension> Tensor<D> where bound!(inner D): Sized {
    /// Computes the Hadamard product (aka. element-wise multiplication)
    pub fn hadamard_product(self, rhs: &Self) -> Self {
        self.map_zip_ref(rhs, |i, j| i * j)
    }

    /// Computes the inner product of the given tensors.
    #[inline]
    pub fn dot(&self, b: &Self) -> f32 {
        self.inner.iter().zip(b.inner.iter()).map(|(i, j)| *i * *j).sum()
    }
}

impl<const W: usize, const H: usize> Matrix<W, H> where
    bound!(inner Dim2<W, H>): Sized,
    bound!(Dim2<H, W>): Sized,
{
    /// Transposes the given matrix.
    pub fn transpose(self) -> Matrix<H, W> {
        let mut t = Matrix::new_filled(0.0);

        for y in 0..H {
            for x in 0..W {
                t[[y, x]] = self[[x, y]];
            }
        }

        t
    }
}

macro_rules! convolution {
    (for_in $p:tt $k:tt $pv:tt $kv:tt ; $wsum:tt $self:tt $kernel:tt $($rp:tt $rk:tt $rpv:tt $rkv:tt),*) => {
        for $kv in $pv.saturating_sub($k)..$pv.min($p) {
            $wsum += $self[[$($rkv),*]] * $kernel[[$($rpv - $rkv - 1),*]];
        }
    };
    (for_in $p:tt $k:tt $pv:tt $kv:tt , $($r:tt)*) => {
        for $kv in $pv.saturating_sub($k)..$pv.min($p) {
            convolution!(for_in $($r)*);
        }
    };
    (for_out $stride:expr, $p:tt $k:tt $pv:tt ; $self:tt $kernel:tt $new:tt $i:tt $($rp:tt $rk:tt $rpv:tt $rkv:tt),*) => {
        for $pv in 1..($p + $k + $stride[0] - 1) / $stride[0] {
            let $pv = ($pv - 1) * $stride[0] + 1;
            let mut wsum = 0.0;
            convolution!(for_in $($rp $rk $rpv $rkv),* ; wsum $self $kernel $($rp $rk $rpv $rkv),*);
            $new.inner[$i] = wsum;
            $i += 1;
        }
    };
    (for_out $stride:expr, $p:tt $k:tt $pv:tt , $($r:tt)*) => {
        for $pv in 1..($p + $k + $stride[0] - 1) / $stride[0] {
            let $pv = ($pv - 1) * $stride[0] + 1;
            convolution!(for_out &$stride[1..], $($r)*);
        }
    };
    ($tensor:tt $D:tt $dim:tt $($i:tt $p:tt $k:tt $b:tt $pv:tt $kv:tt),*) => {
        impl<$(const $p: usize),*> $tensor<$($p),*> where bound!($D<$($p),*>): Sized {
            /// Computes the convolution with padded 0.
            ///
            /// # Stride
            /// The `stride` parameter controls the amount that `kernel` moves in each element.
            #[doc = concat!("Note that it panics when any elememt in `stride[..", $dim, "]` is 0")]
            #[doc = concat!("or `stride.len() < ", $dim, "`. If unsure, use `&[1; ", $dim, "]`")]
            ///
            /// # Note
            /// Each dimension of the output must be `ceil((self + kernel) / stride) - 1` or else
            /// there will be a runtime panic. It is not checked during compile time due to Rust
            /// type bound limits.
            pub fn convolution<$(const $k: usize, const $b: usize),*>(
                &self,
                kernel: &$tensor<$($k),*>,
                stride: &[usize],
            ) -> $tensor<$($b),*> where
                bound!(inner $D<$($k),*>): Sized,
                bound!(inner $D<$($b),*>): Sized,
            {
                $(assert_eq!($b, ($p + $k + stride[$i] - 1) / stride[$i] - 1);)*

                let mut new = Tensor::new_filled(0.0);
                let mut i = 0;

                convolution!(for_out stride, $($p $k $pv),*; self kernel new i $($p $k $pv $kv),*);

                new
            }
        }
    };
}

convolution!(HVector Dim1 1 0 W KW BW w kw);
convolution!(Matrix Dim2 2 0 W KW BW w kw, 1 H KH BH h kh);
convolution!(Tensor3 Dim3 3 0 W KW BW w kw, 1 H KH BH h kh, 2 D KD BD d kd);
convolution!(Tensor4 Dim4 4 0 X KX BX x kx, 1 Y KY BY y ky, 2 Z KZ BZ z kz, 3 W KW BW w kw);
convolution!(Tensor5 Dim5 5 0 X KX BX x kx, 1 Y KY BY y ky, 2 Z KZ BZ z kz, 3 W KW BW w kw, 4 V KV BV v kv);
convolution!(Tensor6 Dim6 6 0 X KX BX x kx, 1 Y KY BY y ky, 2 Z KZ BZ z kz, 3 W KW BW w kw, 4 V KV BV v kv, 5 U KU BU u ku);

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
fn add_assi() {
    let mut a = vector!(3 [5.0, 6.0, 2.0]);
    let b = vector!(3 [1.0, 1.0, 1.0]);

    a += &b;
    assert_eq!(a, vector!(3 [6.0, 7.0, 3.0]));

    a += 1.0;
    assert_eq!(a, vector!(3 [7.0, 8.0, 4.0]));
}

#[cfg(test)]
#[test]
fn convolution() {
    let a = hvector!(8 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    let b = hvector!(3 [0.5, 0.2, 0.3]);
    assert_eq!(a.convolution(&b, &[1]), hvector!(10 [0.5, 0.7, 1.0, 0.5, 0.3, 0.0, 0.5, 0.7, 0.5, 0.3]));

    let a = hvector!(8 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    let b = hvector!(4 [0.5, 0.2, 0.3, 0.1]);
    assert_eq!(a.convolution(&b, &[1]), hvector!(11 [0.5, 0.7, 1.0, 0.6, 0.4, 0.1, 0.5, 0.7, 0.5, 0.4, 0.1]));

    let a = hvector!(10 [3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let b = hvector!(5 [0.5, 0.2, 0.3, 0.1, 0.2]);
    assert_eq!(a.convolution(&b, &[1]), hvector!(14 [1.5, 1.1, 1.6, 0.8, 1.0, 0.3, 0.7, 0.7, 0.5, 0.9, 0.5, 0.5, 0.1, 0.2]));

    let a = hvector!(8 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    let b = hvector!(3 [0.5, 0.2, 0.3]);
    assert_eq!(a.convolution(&b, &[2]), hvector!(5 [0.5, 1.0, 0.3, 0.5, 0.5]));
}
