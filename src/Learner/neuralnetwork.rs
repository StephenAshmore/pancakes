use Tensor::Rank2Tensor;
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

    pub fn validate(&mut self, global_inputs: u64)
    {
        // Compute inputs for each layer based on the first inputs (global_inputs)
        // and then subsequent layers based on the total output of the previous layer.
        self.m_inputs = global_inputs;
        let mut current_inputs = global_inputs;
        let mut current_outputs = 0;

        for i in 0..self.m_blocks.len() {
            current_outputs = 0;
            for j in 0..self.m_blocks[i].len() {
                self.m_blocks[i as usize][j as usize].setInputs(current_inputs);
                current_outputs += self.m_blocks[i as usize][j as usize].outputs();
            }
            current_inputs = current_outputs;
        }
        self.m_outputs = current_outputs;

        self.m_ready = true;
    }

    // TRAIN method for Neural network
    pub fn train(&mut self, features: &Rank2Tensor, labels: &Rank2Tensor)
    {
        // iterate until fully trained

        // randomly iterate through training set

        // for row in 0..features.size()
        // {

        // }
    }

    pub fn test() -> bool {
        let mut nn = NeuralNetwork::new(Box::new(GradientDescent::new(Some(0.0001))) as Box<Optimizer>);
        // Not the most pleasant syntax for moving a trait object.
        nn.add(Box::new(Layer::new(4, TanH::new())) as Box<Differentiable>);
        //nn.concat(Box::new(Layer::new(6, Identity::new())) as Box<Differentiable>);

        let mut input = Rank1Tensor::new(2);
        input[0] = 2.0;
        input[1] = 3.0;

        let mut label = Rank1Tensor::new(4);
        label[0] = 1.0; label[1] = 1.0; label[2] = 2.0; label[3] = 1.0;

        let mut prediction = Rank1Tensor::new(4);

        nn.forward(&input, &mut prediction);

        println!("Prediction: {:?}", prediction);
        println!("The Target: {:?}", label);

        true
    }
}

impl Differentiable for NeuralNetwork {
    fn inputs(&self) -> u64
    {
        self.m_inputs
    }

    fn outputs(&self) -> u64
    {
        self.m_outputs
    }

    fn setInputs(&mut self, new_inputs: u64)
    {
        self.m_inputs = new_inputs;
        self.validate(new_inputs);
    }

    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor) {
        if ( !self.m_ready ) {
            self.validate(input.size());
        }
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
        assert!(self.m_inputs != 0 && input.size() == self.m_inputs,
            "The input Rank1Tensor must be the same size as the number of inputs in this layer!");


        // actually go through the layers and stuff:
        self.m_blocks[0][0].forward(input, prediction);

    }

    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor) {
        // actually go through the layers and stuff:

    }

    fn forwardBatch(&mut self, features: Rank2Tensor) {

    }

    fn backpropBatch(&mut self) {

    }
}
