use Function::functiontraits::*;
use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;

pub struct TanH {

}

impl IsFunction for TanH {
    fn evaluate(&self, value: f64) -> f64 {
        value.tanh()
    }
    fn evaluateRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor) {
        for i in 0..tensor.size() {
            output[i] = tensor[i].tanh();
        }
    }
    fn evaluateRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor){
        for i in 0..tensor.rows() {
            for j in 0..tensor.cols() {
                output[i][j] = tensor[i][j].tanh();
            }
        }
    }

    fn inverse(&self, value: f64) -> f64{
        value.atanh()
    }
    fn inverseRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor){
        for i in 0..tensor.size() {
            output[i] = tensor[i].atanh();
        }
    }
    fn inverseRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor){
        for i in 0..tensor.rows() {
            for j in 0..tensor.cols() {
                output[i][j] = tensor[i][j].atanh();
            }
        }
    }

}
