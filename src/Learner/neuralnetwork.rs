// use Tensor::Rank2Tensor;
// use Tensor::Vector;
use Tensor::Rank1Tensor;
use Learner::Layer;
use Function::functions::*;
use Learner::learnertraits::*;
use Optimizer::*;

// #[derive(Clone)]
pub struct NeuralNetwork {
    m_blocks: Vec<Vec<Box<Differentiable>>>,
    m_layer_count: u64,
    m_optimizer: Box<Optimizer>,
}

impl NeuralNetwork {
    pub fn new(optimizer: Box<Optimizer>) -> NeuralNetwork {
        NeuralNetwork {
            m_blocks: Vec::new(),
            m_layer_count: 0,
            m_optimizer: optimizer,
        }
    }

    pub fn add(&mut self, block: Box<Differentiable>)
    {
        self.m_layer_count += 1;
        let mut new_vec = Vec::new();
        new_vec.push(block);
        self.m_blocks.push(new_vec);
    }

    pub fn concat(&mut self, block: Box<Differentiable>)
    {
        self.m_blocks[(self.m_layer_count-1) as usize].push(block);
    }


// should name this forward?
    pub fn predict(&mut self, input: &Rank1Tensor) -> Rank1Tensor {
        // Iterate through Differentiable blocks, give input to the first blocks
        // propagate the blocks' output to the next layer of blocks.

        // going to have some trouble with how
        // First input Stage:

        // for block_number in 0..(self.m_blocks[0 as usize].len()) {
        //     self.m_blocks[0][block_number].forward(input);
        // }

        Rank1Tensor::new(20)
    }

    pub fn test() -> bool {
        let mut nn = NeuralNetwork::new(Box::new(GradientDescent::new(0.0001)) as Box<Optimizer>);
        // Not the most pleasant syntax for moving a trait object.
        nn.add(Box::new(Layer::new(2, 4, TanH::new())) as Box<Differentiable>);
        nn.concat(Box::new(Layer::new(2, 6, Identity::new())) as Box<Differentiable>);

        true
    }
}
