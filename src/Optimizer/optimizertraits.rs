use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;
use Function::functiontraits::IsFunction;

pub trait Optimizer {
    fn optimize(&mut self, weights: &mut Rank2Tensor, gradient: &Rank1Tensor);
}
