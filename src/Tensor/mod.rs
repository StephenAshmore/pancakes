pub mod Rank1Tensor;
pub mod Rank2Tensor;
// use Tensor::Rank1Tensor;
// pub use Tensor::Rank1Tensor::Rank1Tensor;


pub fn TensorTest() {
    assert!(Rank1Tensor::test(), "Tensor Rank One Test Failed.");

    // assert!(Rank2Tensor::test(), "Tensor Rank Two Test Failed.");
}
