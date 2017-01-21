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
    fn new(inputs: u64, outputs: u64, activation: T) -> Layer<T> {
        assert!((inputs > 0 && outputs > 0), "Inputs and outputs must be 1 or more.");

        Layer {
            m_weights: Rank2Tensor::new(outputs, inputs),
            m_bias: Rank1Tensor::new(outputs),
            m_inputs: inputs,
            m_outputs: outputs,
            m_activation: activation,
            m_net_input: Rank1Tensor::new(outputs),
            m_blame: Rank1Tensor::new(outputs),
        }
    }
}

impl<T: IsFunction> Differentiable for Layer<T> {
    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor) {
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
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
