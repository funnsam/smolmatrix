use core::ops::{Index, IndexMut};

use crate::*;

pub trait VectorDim: Dimension {
    const SIZE: usize;
}

#[allow(private_bounds)]
pub trait VectorDim2: Seal + VectorDim {}
#[allow(private_bounds)]
pub trait VectorDim3: Seal + VectorDim {}
#[allow(private_bounds)]
pub trait VectorDim4: Seal + VectorDim {}
#[allow(private_bounds)]
pub trait VectorDim5: Seal + VectorDim {}
#[allow(private_bounds)]
pub trait VectorDim6: Seal + VectorDim {}

macro_rules! dim {
    ($name:tt $($one:tt)*) => {
        impl<const S: usize> VectorDim for $name<$($one,)* S> {
            const SIZE: usize = S;
        }

        impl VectorDim2 for $name<$($one,)* 2> {}
        impl VectorDim3 for $name<$($one,)* 3> {}
        impl VectorDim4 for $name<$($one,)* 4> {}
        impl VectorDim5 for $name<$($one,)* 5> {}
        impl VectorDim6 for $name<$($one,)* 6> {}
    };
}
dim!(Dim1);
dim!(Dim2 1);
dim!(Dim3 1 1);
dim!(Dim4 1 1 1);
dim!(Dim5 1 1 1 1);
dim!(Dim6 1 1 1 1 1);

impl<D: VectorDim> Index<usize> for Tensor<D> where bound!(inner D): Sized {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<D: VectorDim> IndexMut<usize> for Tensor<D> where bound!(inner D): Sized {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<D: VectorDim> Tensor<D> where bound!(D): Sized {
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

    /// Computes the length from `self` to `[0, …]`.
    #[inline]
    #[cfg(feature = "std")]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Computes the squared length from `self` to `[0, …]`.
    #[inline]
    pub fn length_squared(&self) -> f32 {
        let mut acc = 0.0;

        for i in 0..D::SIZE {
            acc += self[i] * self[i];
        }

        acc
    }

    #[inline]
    #[cfg(feature = "std")]
    pub fn unit(self) -> Self {
        let len = self.length();
        self / len
    }
}

impl<D: VectorDim3> Tensor<D> where bound!(D): Sized {
    #[inline]
    pub fn cross(&self, b: &Self) -> Self {
        let mut new = Self::new_filled(0.0);

        new[0] = self[1] * b[2] - self[2] * b[1];
        new[1] = self[2] * b[0] - self[0] * b[2];
        new[2] = self[0] * b[1] - self[1] * b[0];

        new
    }
}

impl<const S: usize> Vector<S> where bound!(Dim2<1, S>): Sized {
    #[inline]
    pub fn into_horizontal(self) -> HVector<S> where bound!(Dim1<S>): Sized {
        // SAFETY: a tensor and it's transposed variant are the same size
        unsafe {
            HVector { inner: *core::mem::transmute::<&[f32; <Dim2<1, S> as Dimension>::NUM_ELEMENTS], &[f32; <Dim1<S> as Dimension>::NUM_ELEMENTS]>(&self) }
        }
    }
}

impl<const S: usize> HVector<S> where bound!(Dim1<S>): Sized {
    #[inline]
    pub fn into_vertical(self) -> Vector<S> where bound!(Dim2<1, S>): Sized {
        // SAFETY: a tensor and it's transposed variant are the same size
        unsafe {
            Vector { inner: *core::mem::transmute::<&[f32; <Dim1<S> as Dimension>::NUM_ELEMENTS], &[f32; <Dim2<1, S> as Dimension>::NUM_ELEMENTS]>(&self) }
        }
    }
}

#[cfg(test)]
#[test]
fn hvector_index() {
    let mut a = HVector::<4>::new_filled(0.0);

    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 4.0;
    assert_eq!(a, HVector { inner: [1.0, 2.0, 3.0, 4.0] });

    *a.x_mut() = 5.0;
    *a.y_mut() = 6.0;
    *a.z_mut() = 7.0;
    *a.w_mut() = 8.0;
    assert_eq!(a.x(), 5.0);
    assert_eq!(a.y(), 6.0);
    assert_eq!(a.z(), 7.0);
    assert_eq!(a.w(), 8.0);
    assert_eq!(*a.x_ref(), 5.0);
    assert_eq!(*a.y_ref(), 6.0);
    assert_eq!(*a.z_ref(), 7.0);
    assert_eq!(*a.w_ref(), 8.0);
    assert_eq!(a, HVector { inner: [5.0, 6.0, 7.0, 8.0] });
}

#[cfg(test)]
#[test]
fn vector_index() {
    let mut a = Vector::<4>::new_filled(0.0);

    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 4.0;
    assert_eq!(a, Vector { inner: [1.0, 2.0, 3.0, 4.0] });

    *a.x_mut() = 5.0;
    *a.y_mut() = 6.0;
    *a.z_mut() = 7.0;
    *a.w_mut() = 8.0;
    assert_eq!(a.x(), 5.0);
    assert_eq!(a.y(), 6.0);
    assert_eq!(a.z(), 7.0);
    assert_eq!(a.w(), 8.0);
    assert_eq!(*a.x_ref(), 5.0);
    assert_eq!(*a.y_ref(), 6.0);
    assert_eq!(*a.z_ref(), 7.0);
    assert_eq!(*a.w_ref(), 8.0);
    assert_eq!(a, Vector { inner: [5.0, 6.0, 7.0, 8.0] });
}

#[cfg(test)]
#[test]
fn vector3_cross() {
    let a = vector!(3 [5.0, 6.0, 2.0]);
    let b = vector!(3 [1.0, 1.0, 1.0]);

    assert_eq!(a.cross(&b), vector!(3 [4.0, -3.0, -1.0]));
}
