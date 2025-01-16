use core::ops::{Index, IndexMut};

use crate::*;

pub trait VectorDim: Dimension {}
pub trait VectorDim2: VectorDim {}
pub trait VectorDim3: VectorDim {}
pub trait VectorDim4: VectorDim {}
pub trait VectorDim5: VectorDim {}
pub trait VectorDim6: VectorDim {}

macro_rules! dim {
    ($name:tt $($one:tt)*) => {
        impl<const S: usize> VectorDim for $name<$($one,)* S> where [f32; 1 $(* $one)* * S]: Sized {
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

impl<D: VectorDim> Index<usize> for Tensor<D> {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner.as_ref()[index]
    }
}

impl<D: VectorDim> IndexMut<usize> for Tensor<D> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner.as_mut()[index]
    }
}

impl<D: VectorDim> Tensor<D> {
    /// Get the 1st component.
    pub fn x(&self) -> f32 { self[0] }
    /// Get the 2nd component.
    pub fn y(&self) -> f32 { self[1] }
    /// Get the 3rd component.
    pub fn z(&self) -> f32 { self[2] }
    /// Get the 4th component.
    pub fn w(&self) -> f32 { self[3] }

    /// Get the 1st component by reference.
    pub fn x_ref(&self) -> &f32 { &self[0] }
    /// Get the 2nd component by reference.
    pub fn y_ref(&self) -> &f32 { &self[1] }
    /// Get the 3rd component by reference.
    pub fn z_ref(&self) -> &f32 { &self[2] }
    /// Get the 4th component by reference.
    pub fn w_ref(&self) -> &f32 { &self[3] }

    /// Get the 1st component by a mutable reference.
    pub fn x_mut (&mut self) -> &mut f32 { &mut self[0] }
    /// Get the 2nd component by a mutable reference.
    pub fn y_mut (&mut self) -> &mut f32 { &mut self[1] }
    /// Get the 3rd component by a mutable reference.
    pub fn z_mut (&mut self) -> &mut f32 { &mut self[2] }
    /// Get the 4th component by a mutable reference.
    pub fn w_mut (&mut self) -> &mut f32 { &mut self[3] }

    /// Computes the length from `self` to `[0, …]`.
    #[inline]
    #[cfg(feature = "std")]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Computes the squared length from `self` to `[0, …]`.
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.inner.as_ref().iter().map(|i| i * i).sum()
    }

    /// Computes the normalized vector of `self` where [`Self::length`] is ≈1.
    #[inline]
    #[cfg(feature = "std")]
    pub fn unit(self) -> Self {
        let len = self.length();
        self / len
    }
}

impl<D: VectorDim3> Tensor<D> {
    /// Computes the cross product between 2 vectors of length 3.
    #[inline]
    pub fn cross(&self, b: &Self) -> Self {
        let mut new = Self::new_filled(0.0);

        new[0] = self[1] * b[2] - self[2] * b[1];
        new[1] = self[2] * b[0] - self[0] * b[2];
        new[2] = self[0] * b[1] - self[1] * b[0];

        new
    }
}

impl<const S: usize> Vector<S> where [f32; 1 * 1 * S]: Sized {
    /// Transpose `self` into a [horizontal vector](HVector).
    #[inline]
    pub fn into_horizontal(self) -> HVector<S> where
        [f32; 1 * S]: Sized,
        Dim1<S>: SameSized<Dim2<1, S>>,
    {
        self.reshape()
    }
}

impl<const S: usize> HVector<S> where [f32; 1 * S]: Sized {
    /// Transpose `self` into a [vertical vector](Vector).
    #[inline]
    pub fn into_vertical(self) -> Vector<S> where
        [f32; 1 * 1 * S]: Sized,
        Dim2<1, S>: SameSized<Dim1<S>>,
    {
        self.reshape()
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
#[should_panic]
fn hvector_index_panics() {
    let a = HVector::<3>::new_filled(0.0);
    a[3];
}

#[cfg(test)]
#[test]
#[should_panic]
fn hvector_index_fn_panics() {
    let a = HVector::<3>::new_filled(0.0);
    a.w();
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
#[should_panic]
fn vector_index_panics() {
    let a = Vector::<3>::new_filled(0.0);
    a[3];
}

#[cfg(test)]
#[test]
#[should_panic]
fn vector_index_fn_panics() {
    let a = Vector::<3>::new_filled(0.0);
    a.w();
}

#[cfg(test)]
#[test]
fn vector3_cross() {
    let a = vector!(3 [5.0, 6.0, 2.0]);
    let b = vector!(3 [1.0, 1.0, 1.0]);

    assert_eq!(a.cross(&b), vector!(3 [4.0, -3.0, -1.0]));
}
