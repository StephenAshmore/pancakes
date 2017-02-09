use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;

pub trait IsFunction {
    fn evaluate(&self, value: f64) -> f64;
    fn evaluateRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn evaluateRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);

    fn inverse(&self, value: f64) -> f64;
    fn inverseRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn inverseRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);

    fn derivative(&self, value: f64) -> f64;
    fn derivativeRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn derivativeRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);
}

pub trait IsCostFunction {
    fn evaluate(&self, target: f64, prediction: f64) -> f64;
    fn evaluateRank1(&self, target: &Rank1Tensor, prediction: &Rank1Tensor) -> Rank1Tensor;
    fn evaluateRank2(&self, target: &Rank2Tensor, prediction: &Rank2Tensor) -> Rank2Tensor;
}
