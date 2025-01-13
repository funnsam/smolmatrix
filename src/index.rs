use core::ops::{Index, IndexMut};
use crate::*;

impl<D: Dimension> Tensor<D> where bound!(D): Sized {
    #[inline]
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
}

impl<D: Dimension> Index<[usize; D::ORDER]> for Tensor<D> where bound!(D): Sized {
    type Output = f32;

    #[inline]
    fn index(&self, index: [usize; D::ORDER]) -> &Self::Output {
        &self.inner[Self::index_of(index)]
    }
}

impl<D: Dimension> IndexMut<[usize; D::ORDER]> for Tensor<D> where bound!(D): Sized {
    #[inline]
    fn index_mut(&mut self, index: [usize; D::ORDER]) -> &mut Self::Output {
        &mut self.inner[Self::index_of(index)]
    }
}

impl<const W: usize> Index<usize> for HVector<W> where bound!(Dim1<W>): Sized {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<const W: usize> IndexMut<usize> for HVector<W> where bound!(Dim1<W>): Sized {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<const H: usize> Index<usize> for Vector<H> where bound!(Dim2<1, H>): Sized {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<const H: usize> IndexMut<usize> for Vector<H> where bound!(Dim2<1, H>): Sized {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<const W: usize> HVector<W> where bound!(Dim1<W>): Sized {
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

impl<const H: usize> Vector<H> where bound!(Dim2<1, H>): Sized {
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

#[cfg(test)]
#[test]
fn scalar() {
    let mut a = Scalar::new_filled(0.0);

    a[[]] = 1.0;
    assert_eq!(a, Scalar { inner: [1.0] });
}

#[cfg(test)]
#[test]
fn hvector() {
    let mut a = HVector::<3>::new_filled(0.0);

    a[[0]] = 1.0;
    a[[1]] = 2.0;
    a[[2]] = 3.0;
    assert_eq!(a, HVector { inner: [1.0, 2.0, 3.0] });

    a[0] = 4.0;
    a[1] = 5.0;
    a[2] = 6.0;
    assert_eq!(a, HVector { inner: [4.0, 5.0, 6.0] });

    *a.x_mut() = 7.0;
    *a.y_mut() = 8.0;
    *a.z_mut() = 9.0;
    assert_eq!(a, HVector { inner: [7.0, 8.0, 9.0] });
}

#[cfg(test)]
#[test]
fn vector() {
    let mut a = Vector::<3>::new_filled(0.0);

    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;
    a[[0, 2]] = 3.0;
    assert_eq!(a, Vector { inner: [1.0, 2.0, 3.0] });

    a[0] = 4.0;
    a[1] = 5.0;
    a[2] = 6.0;
    assert_eq!(a, Vector { inner: [4.0, 5.0, 6.0] });

    *a.x_mut() = 7.0;
    *a.y_mut() = 8.0;
    *a.z_mut() = 9.0;
    assert_eq!(a, Vector { inner: [7.0, 8.0, 9.0] });
}

#[cfg(test)]
#[test]
fn matrix() {
    let mut a = Matrix::<2, 2>::new_filled(0.0);

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
