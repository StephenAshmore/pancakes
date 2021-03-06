use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;

pub trait IsFunction {
    fn evaluate(&self, value: f64) -> f64;
    fn evaluate_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn evaluate_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);

    fn inverse(&self, value: f64) -> f64;
    fn inverse_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn inverse_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);

    fn derivative(&self, value: f64) -> f64;
    fn derivative_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor);
    fn derivative_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor);
}

pub trait IsCostFunction {
    fn evaluate(&self, target: f64, prediction: f64) -> f64;
    fn evaluate_rank1(&self, target: &Rank1Tensor, prediction: &Rank1Tensor) -> Rank1Tensor;
    fn evaluate_rank2(&self, target: &Rank2Tensor, prediction: &Rank2Tensor) -> Rank2Tensor;
}
