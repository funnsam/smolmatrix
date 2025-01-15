<!-- updated by cargo-release -->

# Unreleased
- Sealed `Dimension`
- Added `Tensor::convolution`, `Tensor::reshape` and `SameSized`
- Probably more that I forgot

# 0.2.1
- Added `bounds!` macro for `where` clauses on generic dimensions
- Fixed documentation

# 0.2.0
- There are now `Tensor` instead of `Matrix`
- Added the Hadamard operation to tensors
- `serde` feature now uses derive
- Flattened `.inner` arrays
- Added `Dimension` trait
- Unfortunately generic things for tensors require `where [f32; D::NUM_ELEMENTS]: Sized`

# 0.1.9
- Added `serde` feature
- Added `FromIterator` for `Matrix`

# 0.1.8
- Added `x/y/z/w(_ref/mut)` methods to vectors
- Added `vector_swap` macro
- Fixed the `matrix` and `vector` macros requiring `use`-ing the `Matrix` struct

# 0.1.x
- Basic vector and matrix math
