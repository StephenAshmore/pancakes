pub use self::neuralnetwork::NeuralNetwork;
// pub use self::neuralblock::NeuralBlock;
pub use self::layer::*;
pub use self::learnertraits::*;
pub use self::activation_layer::*;

mod neuralnetwork;
// mod neuralblock;
mod layer;
mod learnertraits;
mod activation_layer;

pub fn Test() {
    // assert!(NeuralBlock::test(), "Neural Block Test Failed!");


    assert!(NeuralNetwork::test(), "Neural Network Test Failed!");

}
