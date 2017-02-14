use Tensor::Rank2Tensor;
use Tensor::Vector;
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
    m_layer_output_counts: Vector<u64>,
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
            m_layer_output_counts: Vector::new(),
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
            self.m_layer_output_counts.push(current_outputs);
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

    pub fn output_at_layer_count(&self, layer_number: u64) -> u64
    {
        assert!(self.m_ready, "You cannot get the output at a layer without first validating the neural network!"); // this assert may change to allow this function to call validate.

        assert!((layer_number >= 0 && layer_number < self.m_layer_count), "You cannot get the layer count for a layer that does not exist.");

        self.m_layer_output_counts[layer_number]
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
        if !self.m_ready {
            self.validate(input.size());
        }
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
        assert!(self.m_inputs != 0 && input.size() == self.m_inputs,
            "The input Rank1Tensor must be the same size as the number of inputs in this layer!");


        // actually go through the layers and stuff:
        // give the correct input to the next layer:
        let mut current_input = Rank1Tensor::new(input.size());
        current_input.copy(input);
        for i in 0..self.m_layer_count {
            let mut current_output = Rank1Tensor::new(self.output_at_layer_count(i));

            // forward into each block with the current input:
            let mut start_position = 0;
            for j in 0..self.m_blocks[i as usize].len() {
                // forward input here:
                let block_output_count = self.m_blocks[i as usize][j as usize].outputs();
                let mut block_output = Rank1Tensor::new(block_output_count);
                self.m_blocks[i as usize][j as usize].forward(&current_input, &mut block_output);
                current_output.slice_from(&block_output, start_position);
                start_position += block_output_count;
            }
            current_input.copy(&current_output);
        }
        prediction.copy(&current_input);
    }

    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor) {
        assert!(self.m_ready == true,
            "You cannot call backprop on a neural network that has not been validated first.");

        assert!(previous_error.size() == self.m_outputs,
            "The previous error passed to a neural network must have the same size as the number of outputs in the neural network.");

        // actually go through the layers and stuff:
        // give the correct input to the next layer:
        let mut current_error = Rank1Tensor::new(previous_error.size());
        current_error.copy(previous_error);

        for i in self.m_layer_count-1..0 {
            let mut error_start_position = 0; let mut next_error_start_position = 0;
            let mut total_layer_inputs = 0;
            for k in 0..self.m_blocks[i as usize].len() { total_layer_inputs += self.m_blocks[i as usize][k as usize].inputs(); }
            let mut next_block_error = Rank1Tensor::new(total_layer_inputs);
            for j in 0..self.m_blocks[i as usize].len() {
                let mut single_block_error = Rank1Tensor::new(self.m_blocks[i as usize][j as usize].outputs());
                single_block_error.copy_slice(&current_error, error_start_position, None);
                let mut next_single_block_error = Rank1Tensor::new(self.m_blocks[i as usize][j as usize].inputs());

                // give current error to this block and store its backpropagated error
                self.m_blocks[i as usize][j as usize].backprop(&single_block_error, &mut next_single_block_error);

                next_block_error.slice_from(&next_single_block_error, next_error_start_position);
                next_error_start_position += next_single_block_error.size();

                error_start_position += self.m_blocks[i as usize][j as usize].outputs();
            }
            current_error.copy(&next_block_error);
        }

        error.copy(&current_error);
    }

    fn forwardBatch(&mut self, features: Rank2Tensor) {

    }

    fn backpropBatch(&mut self) {

    }
}
