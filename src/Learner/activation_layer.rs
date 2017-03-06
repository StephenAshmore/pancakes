use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Function::functiontraits::*;
use Learner::learnertraits::*;
use Optimizer::optimizertraits::Optimizer;

use rand::Rng;
use rand;
use rand::distributions::{Range, IndependentSample};

pub struct Activation_Layer<T: IsFunction> {
    m_inputs: u64,
    m_outputs: u64,
    m_activation: T,
    m_net_input: Rank1Tensor,
    m_net_output: Rank1Tensor,
    m_gradient: Rank1Tensor,
}

impl<T: IsFunction> Activation_Layer<T> {
    pub fn new(neurons: u64, activation: T) -> Activation_Layer<T> {
        assert!(neurons > 0, "Inputs and outputs must be 1 or more.");

        Activation_Layer {
            m_inputs: 1,
            m_outputs: neurons,
            m_activation: activation,
            m_net_input: Rank1Tensor::new(neurons),
            m_net_output: Rank1Tensor::new(neurons),
            m_gradient: Rank1Tensor::new(neurons),
        }
    }
}

impl<T: IsFunction> Differentiable for Activation_Layer<T> {
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
    }

    fn set_weights(&mut self, weights: &Vec<Rank2Tensor>)
    {
        // do nothing for activation layers.
        // Maybe in the future we will implement activation functions that require weights
    }

    fn first_gradient(&self) -> Rank1Tensor
    {
        self.m_gradient.clone()
    }

    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor) {
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
        assert!(self.m_inputs != 0 && input.size() == self.m_inputs,
            "The input Rank1Tensor must be the same size as the number of inputs in this layer!");

        self.m_net_input.copy(&input);
        self.m_activation.evaluateRank1(&self.m_net_input, prediction);
    }

    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor) {
        // unsquash can be done all in one step??
        let mut unsquash = Rank1Tensor::new(self.m_outputs);
        // unsquash net input, and multiply by the previous_error:
        self.m_activation.derivativeRank1(&self.m_net_input, &mut unsquash);

        unsquash.multiply(previous_error, &mut self.m_gradient);

        // this will insure the error for the next layer is properly set.
        // we no longer need to have access to the downstream layer's weights.
        error.copy(&self.m_gradient);
    }

    fn update(&mut self, optimizer: &mut Box<Optimizer>)
    {
        // do nothing
    }


    fn forwardBatch(&mut self, features: Rank2Tensor) {

    }

    fn backpropBatch(&mut self) {

    }

}
