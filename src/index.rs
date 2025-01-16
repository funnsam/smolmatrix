use crate::*;

impl<D: Dimension> Tensor<D> {
    /// Gets the index of a specific element to `Self::inner`.
    #[inline]
    pub fn index_of(index: &[usize]) -> usize {
        let mut i = 0;
        let mut m = 1;

        for (di, (idx, dim)) in index.into_iter().zip(D::DIMENSIONS.into_iter()).enumerate() {
            assert!(idx < dim, "out of bounds in dimension number {di} ({idx} >= {dim})");

            i += idx * m;
            m *= *dim;
        }

        i
    }
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
}

#[cfg(test)]
#[test]
#[should_panic]
fn hvector_panics() {
    let a = HVector::<3>::new_filled(0.0);
    a[[3]];
}

#[cfg(test)]
#[test]
fn vector() {
    let mut a = Vector::<3>::new_filled(0.0);

    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;
    a[[0, 2]] = 3.0;
    assert_eq!(a, Vector { inner: [1.0, 2.0, 3.0] });
}

#[cfg(test)]
#[test]
#[should_panic]
fn vector_panics() {
    let a = Vector::<3>::new_filled(0.0);
    a[[0, 3]];
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
#[should_panic]
fn matrix_panics_x() {
    let a = Matrix::<2, 2>::new_filled(0.0);
    a[[2, 0]];
}

#[cfg(test)]
#[test]
#[should_panic]
fn matrix_panics_y() {
    let a = Matrix::<2, 2>::new_filled(0.0);
    a[[0, 2]];
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

#[cfg(test)]
#[test]
#[should_panic]
fn tensor3_panics_x() {
    let a = Tensor3::<2, 2, 2>::new_filled(0.0);
    a[[2, 0, 0]];
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor3_panics_y() {
    let a = Tensor3::<2, 2, 2>::new_filled(0.0);
    a[[0, 2, 0]];
}

#[cfg(test)]
#[test]
#[should_panic]
fn tensor3_panics_z() {
    let a = Tensor3::<2, 2, 2>::new_filled(0.0);
    a[[0, 0, 2]];
}
