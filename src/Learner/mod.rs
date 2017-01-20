pub use self::neuralnetwork::NeuralNetwork;
// pub use self::layer::*;

mod neuralnetwork;

pub fn Test() {
    assert!(NeuralNetwork::test(), "Neural Network Test Failed!");



}
