use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;
use Function::functiontraits::IsFunction;

pub trait Optimizer {
    fn optimize(&mut self, weights: &mut Rank2Tensor, bias: &mut Rank1Tensor, input: &mut Rank1Tensor, gradient: &Rank1Tensor);
}
