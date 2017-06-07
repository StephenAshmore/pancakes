use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Function::functiontraits::*;
use Learner::learnertraits::*;
use Optimizer::optimizertraits::Optimizer;

use rand::Rng;
use rand;
use rand::distributions::{Range, IndependentSample};

pub struct Layer {
    m_weights: Rank2Tensor,
    m_bias: Rank1Tensor,
    m_inputs: u64,
    m_outputs: u64,
    m_input: Rank1Tensor,
    m_net_input: Rank1Tensor,
    m_net_output: Rank1Tensor,
    m_gradient: Rank1Tensor,
}

impl Layer {
    pub fn new(neurons: u64) -> Layer {
        assert!(neurons > 0, "Inputs and outputs must be 1 or more.");

        Layer {
            m_weights: Rank2Tensor::new(neurons, 1),
            m_bias: Rank1Tensor::new(neurons),
            m_inputs: 1,
            m_outputs: neurons,
            m_input: Rank1Tensor::new(1),
            m_net_input: Rank1Tensor::new(neurons),
            m_net_output: Rank1Tensor::new(neurons),
            m_gradient: Rank1Tensor::new(neurons),
        }
    }

    pub fn weights(&mut self) -> &mut Rank2Tensor
    {
        &mut self.m_weights
    }
    pub fn print_weights(&self)
    {
        println!("Layer Weights: {:?}", self.m_weights);
    }
}

impl Differentiable for Layer {
    fn inputs(&self) -> u64
    {
        self.m_inputs
    }

    fn outputs(&self) -> u64
    {
        self.m_outputs
    }

    fn set_inputs(&mut self, new_inputs: u64)
    {
        self.m_inputs = new_inputs;
        self.m_weights.resize_columns(self.m_inputs);
        self.m_input.resize(self.m_inputs);

        // initialize weights with small random values for a default layer:
        let mut rng = rand::thread_rng();
        let range = Range::new(0.0, new_inputs as f64);

        for i in 0..self.m_weights.rows() {
            for j in 0..self.m_weights.cols() {
                self.m_weights[i][j] = 0.01 * range.ind_sample(&mut rng);
            }
        }

        // biases are initialized with zeroes, lets do this explicitely here:
        self.m_bias.fill(0.0);
    }

    fn set_weights(&mut self, weights: &Vec<Rank2Tensor>)
    {
        assert!(weights.len() > 1,
                "You can't set weights using an empty vector, you must include the bias as the second Rank2Tensor.");
        assert!(weights[0].rows() == self.m_weights.rows(),
                "set_weights expected to have already had the layer be resized appropriately to the data. Make sure that your weights are the correct size.");
        assert!(weights[0].cols() == self.m_weights.cols(),"Set the weights expects to have already had the layer be resized appropriately to the data. Make sure that your weights are the correct size.");
        self.m_weights.copy(&weights[0]);
        self.m_bias.copy(&weights[1][0]);
    }

    fn first_gradient(&self) -> Rank1Tensor
    {
        self.m_gradient.clone()
    }

    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor)
    {
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
        assert!(self.m_inputs != 0 && input.size() == self.m_inputs,
            "The input Rank1Tensor must be the same size as the number of inputs in this layer!");

        self.m_input.copy(input);
        self.m_net_input = self.m_weights.multiply_rank1(input);
        self.m_net_input = self.m_net_input.add(&self.m_bias);

        prediction.copy(&self.m_net_input);
    }

    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor)
    {
        self.m_gradient.copy(previous_error);

        error.copy(&self.m_weights.multiply_rank1_transpose(&previous_error));
    }

    fn update(&mut self, optimizer: &mut Box<Optimizer>)
    {
        optimizer.optimize(&mut self.m_weights, &mut self.m_bias, &mut self.m_input, &self.m_gradient);
    }

    fn forward_batch(&mut self, features: Rank2Tensor) {

    }

    fn backprop_batch(&mut self) {

    }

}
