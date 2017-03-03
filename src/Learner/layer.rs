use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Function::functiontraits::*;
use Learner::learnertraits::*;

use rand::Rng;
use rand;
use rand::distributions::{Range, IndependentSample};

pub struct Layer {
    m_weights: Rank2Tensor,
    m_bias: Rank1Tensor,
    m_inputs: u64,
    m_outputs: u64,
    m_net_input: Rank1Tensor,
    m_net_output: Rank1Tensor,
    m_blame: Rank1Tensor,
}

impl Layer {
    pub fn new(neurons: u64) -> Layer {
        assert!(neurons > 0, "Inputs and outputs must be 1 or more.");

        Layer {
            m_weights: Rank2Tensor::new(neurons, 1),
            m_bias: Rank1Tensor::new(neurons),
            m_inputs: 1,
            m_outputs: neurons,
            m_net_input: Rank1Tensor::new(neurons),
            m_net_output: Rank1Tensor::new(neurons),
            m_blame: Rank1Tensor::new(neurons),
        }
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

    fn setInputs(&mut self, new_inputs: u64)
    {
        self.m_inputs = new_inputs;
        println!("Resizing weights to have cols: {}", self.m_inputs);
        self.m_weights.resizeColumns(self.m_inputs);

        // Initialization:
        // This process should be the same for most activation functions. For ReLu,
        // they will need to be their own Differentiable type because the weights
        // will need to be initialized differently for best results.
        // initialize weights:
        let mut rng = rand::thread_rng();
        let range = Range::new(0.0, new_inputs as f64);
        // silly way to initialize the weights because I can't figure out how to get max to work.
        let mag1 = 3.0;
        let mag2 = 1.0 / (new_inputs as f64);
        let mut mag = mag2;
        if ( mag1 > mag2 ) {
            mag = mag1;
        }

        for i in 0..self.m_weights.rows() {
            for j in 0..self.m_weights.cols() {
                self.m_weights[i][j] = range.ind_sample(&mut rng) / (new_inputs as f64).sqrt();
            }
        }

        // biases are initialized with zeroes, lets do this explicitely here:
        self.m_bias.fill(0.0);

        println!("Magnitude: {:?}", mag);
        println!("Weight Matrix: {:?}", self.m_weights);
    }

    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor) {
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
        assert!(self.m_inputs != 0 && input.size() == self.m_inputs,
            "The input Rank1Tensor must be the same size as the number of inputs in this layer!");

        println!("Weights rows: {} cols: {}. Input size: {}", self.m_weights.rows(), self.m_weights.cols(), input.size());
        self.m_net_input = self.m_weights.multiplyRank1(input);
        self.m_net_input = self.m_net_input.add(&self.m_bias);

        prediction.copy(&self.m_net_input);
    }

    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor) {
        error.copy(&self.m_weights.multiplyRank1(&previous_error));
    }

    fn forwardBatch(&mut self, features: Rank2Tensor) {

    }

    fn backpropBatch(&mut self) {

    }

}
