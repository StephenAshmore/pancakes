use Tensor::*;
use Optimizer::optimizertraits::Optimizer;

// Should the error and previous_error here be a Rank1Tensor or Rank2Tensor?

pub trait Differentiable {
    fn forward<'a>(&'a mut self, input: &Rank1Tensor, prediction: &'a mut Rank1Tensor);
    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor);

    fn forwardBatch(&mut self, inputs: Rank2Tensor);
    fn backpropBatch(&mut self);

    fn inputs(&self) -> u64;
    fn outputs(&self) -> u64;

    fn setInputs(&mut self, new_inputs: u64);
    fn set_weights(&mut self, weights: &Vec<Rank2Tensor>);

    fn update(&mut self, optimizer: &mut Box<Optimizer>);
    fn first_gradient(&self) -> Rank1Tensor;
}
