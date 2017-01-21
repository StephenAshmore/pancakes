use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;

pub trait IsFunction {
    fn evaluate(&self, value: f64) -> f64;
    fn evaluateRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn evaluateRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);

    fn inverse(&self, value: f64) -> f64;
    fn inverseRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn inverseRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);
}
