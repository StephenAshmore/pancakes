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
    m_ready: bool,
    m_inputs: u64,
    m_outputs: u64,
}

impl NeuralNetwork {
    pub fn new(optimizer: Box<Optimizer>) -> NeuralNetwork
    {
        NeuralNetwork {
            m_blocks: Vec::new(),
            m_layer_count: 0,
            m_optimizer: optimizer,
            m_ready: false,
            m_inputs: 0,
            m_outputs: 0,
        }
    }

    pub fn add(&mut self, block: Box<Differentiable>)
    {
        self.m_layer_count += 1;
        let mut new_vec = Vec::new();
        new_vec.push(block);
        self.m_blocks.push(new_vec);
        self.m_ready = false;
    }

    pub fn concat(&mut self, block: Box<Differentiable>)
    {
        self.m_blocks[(self.m_layer_count-1) as usize].push(block);
        self.m_ready = false;
    }

    pub fn concatAtPosition(&mut self, block: Box<Differentiable>, pos: u64)
    {
        assert!(pos < self.m_layer_count, "You can only concatenate blocks at a position less than the number of layers in the neural network.");

        self.m_blocks[(pos) as usize].push(block);
        self.m_ready = false;

    }
    // TRAIN method for Neural network
    pub fn train(&mut self, features: &Rank2Tensor, labels: &Rank2Tensor)
    {
        // iterate until fully trained
        
        // randomly iterate through training set

        for row in 0..features.size()
        {

        }
    }

    // FORWARD Method for Neural Network
    pub fn forward(&mut self, input: &Rank1Tensor) -> Rank1Tensor {
        if ( !self.m_ready )
        {
            // lazily compute the inputs and outputs
            // compute total inputs:
            self.m_inputs = 0;
            for i in 0..self.m_blocks[0].size() {
                self.m_inputs = self.m_inputs + self.m_blocks[0][i];
            }
            // compute total outputs:
            self.m_outputs = 0;
            for i in 0..self.m_blocks[m_layer_count as usize].size() {
                self.m_outputs = self.m_outputs + self.m_blocks[m_layer_count as usize][i];
            }

            self.m_ready = true;

            // Iterate through Differentiable blocks, give input to the first blocks
            // propagate the blocks' output to the next layer of blocks.


        }

        assert!(input.size() == m_inputs, "");

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

        let mut in = Rank1Tensor::new(2);


        true
    }
}
