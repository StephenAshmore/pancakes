use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Function::functiontraits::*;
use Learner::learnertraits::*;

pub struct Layer<T: IsFunction> {
    m_weights: Rank2Tensor,
    m_bias: Rank1Tensor,
    m_inputs: u64,
    m_outputs: u64,
    m_activation: T,
    m_net_input: Rank1Tensor,
    m_blame: Rank1Tensor, // blame here is possibly the gradient? Not sure, need to clarify this.
}

impl<T: IsFunction> Layer<T> {
    pub fn new(neurons: u64, activation: T) -> Layer<T> {
        assert!(neurons > 0, "Inputs and outputs must be 1 or more.");

        Layer {
            m_weights: Rank2Tensor::new(neurons, 1),
            m_bias: Rank1Tensor::new(neurons),
            m_inputs: 1,
            m_outputs: neurons,
            m_activation: activation,
            m_net_input: Rank1Tensor::new(neurons),
            m_blame: Rank1Tensor::new(neurons),
        }
    }
}

impl<T: IsFunction> Differentiable for Layer<T> {
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
        self.m_weights.resize(self.m_outputs, self.m_inputs);
    }

    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor) {
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
        assert!(self.m_inputs != 0 && input.size() == self.m_inputs,
            "The input Rank1Tensor must be the same size as the number of inputs in this layer!");
        self.m_net_input = self.m_weights.multiplyRank1(input);
        self.m_net_input = self.m_net_input.add(&self.m_bias);

        self.m_activation.evaluateRank1(&self.m_net_input, prediction);
    }

    // Okay the number of outputs does not equal the number of weights in the downstream layer
    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor) {
        // unsquash can be done all in one step??
        let mut unsquash = Rank1Tensor::new(self.m_outputs);

        self.m_blame = self.m_weights.multiplyRank1(previous_error);

        for i in 0..self.m_outputs {
            self.m_blame[i] = self.m_blame[i] * unsquash[i];
            error[i] = self.m_blame[i];
        }
    }

    fn forwardBatch(&mut self, features: Rank2Tensor) {

    }

    fn backpropBatch(&mut self) {

    }

}
