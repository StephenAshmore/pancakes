use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Function::functiontraits::*;
use Learner::learnertraits::*;

pub struct Layer<T> {
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
        self.m_net_input = self.m_weights.multiplyRank1(input);
    }

    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor) {

    }

    fn forwardBatch(&mut self, features: Rank2Tensor) {

    }

    fn backpropBatch(&mut self) {

    }

}
