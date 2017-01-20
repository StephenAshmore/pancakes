pub use self::neuralnetwork::NeuralNetwork;
// pub use self::neuralblock::NeuralBlock;
pub use self::layer::*;
pub use self::learnertraits::*;

mod neuralnetwork;
// mod neuralblock;
mod layer;
mod learnertraits;

pub fn Test() {
    // assert!(NeuralBlock::test(), "Neural Block Test Failed!");


    assert!(NeuralNetwork::test(), "Neural Network Test Failed!");

}
