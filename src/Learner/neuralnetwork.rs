// use Tensor::Rank2Tensor;
// use Tensor::Vector;
// use Tensor::Rank1Tensor;
use Learner::Layer;
use Function::functions::*;
use Learner::learnertraits::*;

// #[derive(Clone)]
pub struct NeuralNetwork {
    m_blocks: Vec<Vec<Box<Differentiable>>>,
    m_layer_count: u64,
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork {
            m_blocks: Vec::new(),
            m_layer_count: 0,
        }
    }

    pub fn add(&mut self, block: Differentiable)
    where Self: Sized
    {
        self.m_layer_count += 1;
        let mut new_vec = Vec::new();
        new_vec.push(Box::new(block));
        self.m_blocks.push(new_vec);
    }


    pub fn test() -> bool {
        let mut nn = NeuralNetwork::new();
        nn.add(&Layer::new(2, 4, TanH::new()));

        true
    }
}

// Okay, we can use traits to do inheritance.
// For example if we need a block to implement the feed forward method, we can do that with a trait
// probably should have several methods belong to the same trait, like feed forward and backprop
// etc.


// What should the interface be?
/*
let mut network = NeuralNetwork::new();
network.span(); // span
network.add();


*/
