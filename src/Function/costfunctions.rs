use Function::functiontraits::*;
use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;

pub struct SSE{}
impl SSE {
    pub fn new() -> SSE {
        SSE { }
    }
}


// Should return a single SSE number, or a rank1tensor?
impl IsCostFunction for SSE {
    fn evaluate(&self, target: f64, prediction: f64) -> f64 {
        (target - prediction) * (target - prediction)
    }

    fn evaluateRank1(&self, target: &Rank1Tensor, prediction: &Rank1Tensor) -> Rank1Tensor {

    }

    fn evaluateRank2(&self, target: &Rank2Tensor, prediction: &Rank2Tensor) -> Rank2Tensor {

    }
}
