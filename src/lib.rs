#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use core::{borrow::{Borrow, BorrowMut}, fmt, ops::{Deref, DerefMut, Index, IndexMut}};

mod index;
mod ops;
pub mod vector;

trait Seal {}

impl<D: Dimension> Seal for Tensor<D> {}
impl<T, const S: usize> Seal for [T; S] {}

#[allow(private_bounds)]
pub trait SameSized<T>: Seal {}

impl<T, const S: usize> SameSized<[T; S]> for [T; S] {}

impl<D: Dimension, E: Dimension> SameSized<E> for D where
    [f32; D::NUM_ELEMENTS]: SameSized<[f32; E::NUM_ELEMENTS]>,
{}

impl<D: Dimension, E: Dimension + SameSized<D>> SameSized<Tensor<E>> for Tensor<D> {}

#[cfg(feature = "serde")]
pub trait Serde<'a>: serde::Serialize + serde::Deserialize<'a> {}
#[cfg(not(feature = "serde"))]
pub trait Serde<'a> {}

#[cfg(feature = "serde")]
impl<'a, T: serde::Serialize + serde::Deserialize<'a>> Serde<'a> for T {}
#[cfg(not(feature = "serde"))]
impl<'a, T> Serde<'a> for T {}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tensor<D: Dimension> {
    pub inner: D::ArrayForm,
}

impl<D: Dimension> AsRef<D::ArrayForm> for Tensor<D> {
    fn as_ref(&self) -> &D::ArrayForm {
        &self.inner
    }
}

impl<D: Dimension> AsMut<D::ArrayForm> for Tensor<D> {
    fn as_mut(&mut self) -> &mut D::ArrayForm {
        &mut self.inner
    }
}

impl<D: Dimension> Deref for Tensor<D> {
    type Target = D::ArrayForm;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Dimension> DerefMut for Tensor<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Dimension> Tensor<D> {
    /// Create a new [`Tensor`] with values filled from `value`.
    pub fn new_filled(value: f32) -> Self {
        Self::from_iter(core::iter::repeat(value))
    }

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
        self.inner.as_mut().iter_mut().for_each(|i| *i = f(*i));
    }

    pub fn map_zip_ref_in_place<F: Fn(f32, f32) -> f32>(&mut self, r: &Self, f: F) {
        let f = &f;
        self.inner.as_mut()
            .iter_mut()
            .zip(r.inner.as_ref().iter())
            .for_each(|(i, j)| *i = f(*i, *j));
    }

    pub fn reshape<E: Dimension>(self) -> Tensor<E> where
        E: SameSized<D>,
    {
        // SAFETY: D and E are the same size bounded by the where clause
        unsafe {
            Tensor { inner: core::mem::transmute_copy(&core::mem::ManuallyDrop::new(self.inner)) }
        }
    }
}

/// Representing a size and order of a [`Tensor`].
///
/// This trait is sealed so that no possible underlying unsafe code are unsafe.
#[allow(private_bounds)]
pub trait Dimension: fmt::Debug + Clone + PartialEq + Seal + for<'a> Serde<'a> {
    /// The order (or rank) of this dimension
    const ORDER: usize;

    /// The dimensions of this dimension
    const DIMENSIONS: &[usize];

    /// The total number of elements.
    ///
    /// This value should be equal to the product of every element of [`Self::DIMENSIONS`]
    const NUM_ELEMENTS: usize;

    type ArrayForm: Clone
        + AsRef<[f32]>
        + AsMut<[f32]>
        + Borrow<[f32]>
        + BorrowMut<[f32]>
        + fmt::Debug
        + PartialEq
        + Sized
        + for<'a> TryFrom<&'a [f32]>;

    type Indicies: Clone
        + AsRef<[usize]>
        + AsMut<[usize]>
        + Borrow<[usize]>
        + BorrowMut<[usize]>
        + fmt::Debug
        + PartialEq
        + Sized
        + for<'a> TryFrom<&'a [usize]>;
}

macro_rules! dim {
    (conv $fn:tt <=> $tn:tt $($ti:tt),*) => {
        impl<$(const $ti: usize),*> From<Tensor<$fn<$($ti,)* 1>>> for Tensor<$tn<$($ti),*>> where
            $tn<$($ti,)*>: SameSized<$fn<$($ti,)* 1>>,
            $fn<$($ti,)* 1>: Dimension<ArrayForm = [f32; 1 $(* $ti)*]>,
        {
            fn from(value: Tensor<$fn<$($ti,)* 1>>) -> Self {
                value.down_conv()
            }
        }

        impl<$(const $ti: usize),*> From<Tensor<$tn<$($ti),*>>> for Tensor<$fn<$($ti,)* 1>> where
            $fn<$($ti,)* 1>: Dimension<ArrayForm = [f32; 1 $(* $ti)*]> + SameSized<$tn<$($ti),*>>,
        {
            fn from(value: Tensor<$tn<$($ti),*>>) -> Self {
                value.up_conv()
            }
        }

        impl<$(const $ti: usize),*> Tensor<$tn<$($ti),*>> where
            $fn<$($ti,)* 1>: Dimension<ArrayForm = [f32; 1 $(* $ti)*]> + SameSized<$tn<$($ti),*>>,
        {
            /// Convert a tensor up an order.
            ///
            /// # Example
            /// ```ignore
            /// #![allow(incomplete_features)]
            /// #![feature(generic_const_exprs)]
            ///
            /// use smolmatrix::*;
            ///
            /// let a = Scalar::new_filled(1.2).up_conv();
            /// assert_eq!(a, HVector { inner: [1.2] });
            /// ```
            pub fn up_conv(self) -> Tensor<$fn<$($ti,)* 1>> {
                self.reshape()
            }

            /// Join multiple tensors into a tensor with an order higher by copying each element.
            ///
            /// # Example
            /// ```ignore
            /// #![allow(incomplete_features)]
            /// #![feature(generic_const_exprs)]
            ///
            /// use smolmatrix::*;
            ///
            /// let a = HVector::join([
            ///     &hvector!(3 [1.0, 2.0, 3.0]),
            ///     &hvector!(3 [4.0, 5.0, 6.0]),
            ///     &hvector!(3 [7.0, 8.0, 9.0]),
            /// ]);
            /// assert_eq!(a, matrix!(3 x 3
            ///     [1.0, 2.0, 3.0]
            ///     [4.0, 5.0, 6.0]
            ///     [7.0, 8.0, 9.0]
            /// ));
            /// ```
            pub fn join<const N: usize>(list: [&Self; N]) -> Tensor<$fn<$($ti,)* N>> where
                [f32; 1 $(* $ti)* * N]: Sized,
            {
                // SAFETY: every element are copied from the inputs
                unsafe {
                    let mut joined: Tensor<$fn<$($ti,)* N>> = core::mem::MaybeUninit::uninit().assume_init();

                    for (i, e) in list.into_iter().enumerate() {
                        joined.inner[i $(* $ti)*..(i + 1) $(* $ti)*].copy_from_slice(&**e);
                    }

                    joined
                }
            }
        }

        impl<$(const $ti: usize),*> Tensor<$fn<$($ti,)* 1>> where
            $tn<$($ti,)*>: SameSized<$fn<$($ti,)* 1>>,
            $fn<$($ti,)* 1>: Dimension<ArrayForm = [f32; 1 $(* $ti)*]>,
        {
            /// Convert a tensor down an order if its last dimension is 1.
            ///
            /// # Example
            /// ```ignore
            /// #![allow(incomplete_features)]
            /// #![feature(generic_const_exprs)]
            ///
            /// use smolmatrix::*;
            ///
            /// let a = hvector!(1 [1.2]).down_conv();
            /// assert_eq!(a, Scalar { inner: [1.2] });
            /// ```
            pub fn down_conv(self) -> Tensor<$tn<$($ti),*>> {
                self.reshape()
            }
        }
    };
    ($n:tt $t:tt $d:tt $ord:tt $($i:tt),*) => {
        #[doc = concat!("Marker object representing a ", $d, "-dimensional size. This struct")]
        #[doc = "contains a private empty tuple so that it isn't constructable."]
        #[derive(Debug, Clone, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $n<$(const $i: usize),*>(pub(self) ());

        impl<$(const $i: usize),*> Seal for $n<$($i),*> {}

        impl<$(const $i: usize),*> Dimension for $n<$($i),*> where [f32; 1 $(* $i)*]: Sized {
            const ORDER: usize = $d;
            const DIMENSIONS: &[usize] = &[$($i),*];
            const NUM_ELEMENTS: usize = 1 $(* $i)*;

            type ArrayForm = [f32; 1 $(* $i)*];
            type Indicies = [usize; $d];
        }

        impl<$(const $i: usize),*> Index<[usize; $d]> for Tensor<$n<$($i),*>> where
            [f32; 1 $(* $i)*]: Sized,
        {
            type Output = f32;

            #[inline]
            fn index(&self, index: [usize; $d]) -> &Self::Output {
                unsafe { self.inner.as_ref().get_unchecked(Self::index_of(&index)) }
            }
        }

        impl<$(const $i: usize),*> IndexMut<[usize; $d]> for Tensor<$n<$($i),*>> where
            [f32; 1 $(* $i)*]: Sized,
        {
            #[inline]
            fn index_mut(&mut self, index: [usize; $d]) -> &mut Self::Output {
                unsafe { self.inner.as_mut().get_unchecked_mut(Self::index_of(&index)) }
            }
        }

        #[doc = concat!("A ", $d, "-", $ord, " order tensor.")]
        pub type $t<$(const $i: usize),*> = Tensor<$n<$($i),*>>;
    };
}

/// Type alias of a vertical vector.
pub type Vector<const H: usize> = Matrix<1, H>;

dim!(Dim0 Scalar  0 "th");
dim!(Dim1 HVector 1 "st" W);
dim!(Dim2 Matrix  2 "nd" W, H);
dim!(Dim3 Tensor3 3 "rd" W, H, D);
dim!(Dim4 Tensor4 4 "th" X, Y, Z, W);
dim!(Dim5 Tensor5 5 "th" X, Y, Z, W, V);
dim!(Dim6 Tensor6 6 "th" X, Y, Z, W, V, U);

dim!(conv Dim1 <=> Dim0);
dim!(conv Dim2 <=> Dim1 W);
dim!(conv Dim3 <=> Dim2 W, H);
dim!(conv Dim4 <=> Dim3 W, H, D);
dim!(conv Dim5 <=> Dim4 X, Y, Z, W);
dim!(conv Dim6 <=> Dim5 X, Y, Z, W, V);

/// Create a new [`Vector`] from a list of elements.
///
/// # Example
/// ```
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use smolmatrix::*;
///
/// let vector = vector!(3 [1.0, 2.0, 3.0]);
/// assert_eq!(vector[0], 1.0);
/// assert_eq!(vector[1], 2.0);
/// assert_eq!(vector[2], 3.0);
/// ```
#[macro_export]
macro_rules! vector {
    ($size:tt $e:expr) => { $crate::Vector::<$size> { inner: $e } };
}


/// Create a new [`HVector`] from a list of elements.
///
/// # Example
/// ```
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use smolmatrix::*;
///
/// let hvector = hvector!(3 [1.0, 2.0, 3.0]);
/// assert_eq!(hvector[0], 1.0);
/// assert_eq!(hvector[1], 2.0);
/// assert_eq!(hvector[2], 3.0);
/// ```
#[macro_export]
macro_rules! hvector {
    ($size:tt $e:expr) => { $crate::HVector::<$size> { inner: $e } };
}

#[doc(hidden)]
#[macro_export]
macro_rules! matrix_fill {
    ($m:expr, $y:expr, [$($v:expr,)+] $($rest:tt)*) => {{
        $crate::matrix_fill!($m, $y, 0, $($v,)*);
        $crate::matrix_fill!($m, $y + 1, $($rest)*);
    }};
    ($m: expr, $y: expr, $x: expr, $v: expr, $($rest:tt)*) => {{
        $m[[$x, $y]] = $v;
        $crate::matrix_fill!($m, $y, $x + 1, $($rest)*);
    }};

    ($m: expr, $y: expr,) => {};
    ($m: expr, $y: expr, $x: expr,) => {};
}

/// Create a new [`Matrix`] from a list of elements.
///
/// # Example
/// ```
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use smolmatrix::*;
///
/// let matrix = matrix!(2 x 2
///     [1.0, 2.0]
///     [3.0, 4.0]
/// );
/// assert_eq!(matrix[[0, 0]], 1.0);
/// assert_eq!(matrix[[1, 0]], 2.0);
/// assert_eq!(matrix[[0, 1]], 3.0);
/// assert_eq!(matrix[[1, 1]], 4.0);
/// ```
#[macro_export]
macro_rules! matrix {
    ($w:tt x $h:tt $([$($v:expr),* $(,)?])*) => {{
        let mut m = $crate::Matrix::<$w, $h>::new_filled(0.0);

        $crate::matrix_fill!(m, 0, $([$($v,)*])*);

        m
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! _vector_swap {
    ($at:expr, $vector:expr, $orig:expr, $idx:tt $($rest:tt)*) => {{
        $vector[$at] = $orig[$idx];
        $crate::_vector_swap!($at + 1, $vector, $orig, $($rest)*);
    }};
    ($at:expr, $vector:expr, $orig:expr,) => {};
}

/// Performs a parallel vector indicies swap by copying.
///
/// # Example
/// ```
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use smolmatrix::*;
///
/// let original = vector!(3 [1.0, 2.0, 3.0]);
/// let swapped = vector_swap!(original, 2 0 1);
///
/// assert_eq!(swapped, vector!(3 [3.0, 1.0, 2.0]));
/// ```
#[macro_export]
macro_rules! vector_swap {
    ($vector:expr, $($idx:tt)*) => {{
        let og = &$vector;
        let mut v = $vector.clone();
        $crate::_vector_swap!(0, v, og, $($idx)*);
        v
    }};
}

impl<D: Dimension> FromIterator<f32> for Tensor<D> {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut inner: D::ArrayForm = unsafe { core::mem::MaybeUninit::uninit().assume_init() };

        let mut wrote = 0;
        for (i, e) in iter.into_iter().enumerate().take(D::NUM_ELEMENTS) {
            inner.as_mut()[i] = e;
            wrote = i + 1;
        }

        if wrote < D::NUM_ELEMENTS {
            panic!("expected iterator with at least {} items, but only got {wrote} before end of \
            iterator", D::NUM_ELEMENTS);
        }

        Self { inner }
    }
}

impl<const W: usize, const H: usize> fmt::Display for Matrix<W, H> where
    [f32; 1 * W * H]: Sized,
{
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
        for y in 0..H {
            for x in 0..W {
                use core::fmt::Write;

                let mut len = LenOf { len: 0 };
                write!(&mut len, "{}", self.inner[Self::index_of(&[x, y])])?;
                align[x] = align[x].max(len.len);
            }
        }

        let inner_space = align.iter().sum::<usize>() + W + 1;
        writeln!(f, "┌{:<inner_space$}┐", "")?;

        for y in 0..H {
            write!(f, "│ ")?;

            for x in 0..W {
                write!(f, "{:^1$} ", self.inner[Self::index_of(&[x, y])], align[x])?;
            }

            writeln!(f, "│")?;
        }

        writeln!(f, "└{:<inner_space$}┘", "")
    }
}

#[cfg(test)]
#[test]
fn down_conv() {
    let a = hvector!(1 [1.2]).down_conv();
    assert_eq!(a, Scalar { inner: [1.2] });

    let a = matrix!(2 x 1 [1.2, 2.3]).down_conv();
    assert_eq!(a, HVector { inner: [1.2, 2.3] });
}

#[cfg(test)]
#[test]
fn up_conv() {
    let a = Scalar::new_filled(1.2).up_conv();
    assert_eq!(a, HVector { inner: [1.2] });

    let a = hvector!(2 [1.2, 2.3]).up_conv();
    assert_eq!(a, Matrix { inner: [1.2, 2.3] });

    let a = matrix!(2 x 2 [1.0, 2.0] [3.0, 4.0]).up_conv();
    assert_eq!(a, Tensor3 { inner: [1.0, 2.0, 3.0, 4.0] });
}

#[cfg(test)]
#[test]
fn join() {
    let a = HVector::join([
        &hvector!(3 [1.0, 2.0, 3.0]),
        &hvector!(3 [4.0, 5.0, 6.0]),
        &hvector!(3 [7.0, 8.0, 9.0]),
    ]);
    assert_eq!(a, matrix!(3 x 3
        [1.0, 2.0, 3.0]
        [4.0, 5.0, 6.0]
        [7.0, 8.0, 9.0]
    ));
}
