pub use self::rank1Tensor::Rank1Tensor;
pub use self::rank2Tensor::Rank2Tensor;
pub use self::vector::Vector;

mod rank1Tensor;
mod rank2Tensor;
mod vector;

pub fn TensorTest() {
    assert!(Rank1Tensor::test(), "Tensor Rank One Test Failed.");

    assert!(Rank2Tensor::test(), "Tensor Rank Two Test Failed.");
}
