use core::ops::{Index, IndexMut};
use crate::*;

impl<D: Dimension> Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    pub fn index_of(index: [usize; D::ORDER]) -> usize {
        let mut i = 0;
        let mut m = 1;

        for (di, (idx, dim)) in index.into_iter().zip(D::DIMENSIONS.into_iter()).enumerate() {
            assert!(idx < *dim, "out of bounds in dimension number {di} ({idx} >= {dim})");

            i += idx * m;
            m *= *dim;
        }

        i
    }

    pub fn new_filled(value: f32) -> Self {
        Self { inner: [value; D::NUM_ELEMENTS] }
    }
}

impl<D: Dimension> Index<[usize; D::ORDER]> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    type Output = f32;

    fn index(&self, index: [usize; D::ORDER]) -> &Self::Output {
        &self.inner[Self::index_of(index)]
    }
}

impl<D: Dimension> IndexMut<[usize; D::ORDER]> for Tensor<D> where [f32; D::NUM_ELEMENTS]: Sized {
    fn index_mut(&mut self, index: [usize; D::ORDER]) -> &mut Self::Output {
        &mut self.inner[Self::index_of(index)]
    }
}

#[cfg(test)]
#[test]
fn scalar() {
    let mut a = Scalar::new_filled(0.0);
    assert_eq!(a, Scalar { inner: [0.0] });
    a[[]] = 1.0;
    assert_eq!(a, Scalar { inner: [1.0] });
}

#[cfg(test)]
#[test]
fn hvector() {
    let mut a = HVector::<3>::new_filled(0.0);
    assert_eq!(a, HVector { inner: [0.0; 3] });
    a[[0]] = 1.0;
    a[[1]] = 2.0;
    a[[2]] = 3.0;
    assert_eq!(a, HVector { inner: [1.0, 2.0, 3.0] });
}

#[cfg(test)]
#[test]
fn matrix() {
    let mut a = Matrix::<2, 2>::new_filled(0.0);
    assert_eq!(a, Matrix { inner: [0.0; 4] });
    a[[0, 0]] = 1.0;
    a[[1, 0]] = 2.0;
    a[[0, 1]] = 3.0;
    a[[1, 1]] = 4.0;
    assert_eq!(a, Matrix { inner: [1.0, 2.0, 3.0, 4.0] });
}

#[cfg(test)]
#[test]
fn tensor3() {
    let mut a = Tensor3::<2, 2, 2>::new_filled(0.0);
    assert_eq!(a, Tensor { inner: [0.0; 8] });
    a[[0, 0, 0]] = 1.0;
    a[[1, 0, 0]] = 2.0;
    a[[0, 1, 0]] = 3.0;
    a[[1, 1, 0]] = 4.0;
    a[[0, 0, 1]] = 5.0;
    a[[1, 0, 1]] = 6.0;
    a[[0, 1, 1]] = 7.0;
    a[[1, 1, 1]] = 8.0;
    assert_eq!(a, Tensor { inner: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] });
}
