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
        target.sub_square(prediction)
    }

    // evaluateRank2 does not really make sense in the scope of a neural network.
    // Essentially a cost function will take a target vector and a prediction Vector
    // This evaluateRank2 will be used for batch training, so maybe it should be named
    // more appropriately?
    fn evaluateRank2(&self, target: &Rank2Tensor, prediction: &Rank2Tensor) -> Rank2Tensor {
        target.sub_square(prediction)
    }
}
pub struct SimpleDifference{}
impl SimpleDifference {
    pub fn new() -> SimpleDifference {
        SimpleDifference { }
    }
}


// Should return a single SSE number, or a rank1tensor?
impl IsCostFunction for SimpleDifference {
    fn evaluate(&self, target: f64, prediction: f64) -> f64 {
        (target - prediction)
    }

    fn evaluateRank1(&self, target: &Rank1Tensor, prediction: &Rank1Tensor) -> Rank1Tensor {
        target.sub(prediction)
    }

    // evaluateRank2 does not really make sense in the scope of a neural network.
    // Essentially a cost function will take a target vector and a prediction Vector
    // This evaluateRank2 will be used for batch training, so maybe it should be named
    // more appropriately?
    fn evaluateRank2(&self, target: &Rank2Tensor, prediction: &Rank2Tensor) -> Rank2Tensor {
        target.sub(prediction)
    }
}
