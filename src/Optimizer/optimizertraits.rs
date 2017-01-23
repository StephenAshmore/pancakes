use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;

pub trait Optimizer {
    fn optimize(&mut self, input: &Rank2Tensor, output: &mut Rank2Tensor);
    fn optimize1D(&mut self, input: &Rank1Tensor, output: &mut Rank1Tensor);
}
