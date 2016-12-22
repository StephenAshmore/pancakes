// pub mod Vector;
// pub mod Rank1Tensor;
// pub mod Rank2Tensor;
// use Tensor::Rank1Tensor;
pub use self::vector::Vector;
pub use self::rank1Tensor::Rank1Tensor;
pub use self::rank2Tensor::Rank2Tensor;

mod vector;
mod rank1Tensor;
mod rank2Tensor;

pub fn TensorTest() {
    assert!(vector::test(), "Vector Test Failed.");

    assert!(rank1Tensor::test(), "Tensor Rank One Test Failed.");

    assert!(rank2Tensor::test(), "Tensor Rank Two Test Failed.");
}
