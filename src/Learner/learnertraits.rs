use Tensor::*;

// Should the error and previous_error here be a Rank1Tensor or Rank2Tensor?

pub trait Differentiable {
    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor);
    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor);

    fn forwardBatch(&mut self, inputs: Rank2Tensor);
    fn backpropBatch(&mut self);

}
