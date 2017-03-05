use Function::functiontraits::*;
use Tensor::Rank1Tensor;
use Tensor::Rank2Tensor;

#[derive(Debug, Clone)]
pub struct TanH{}
impl TanH {
    pub fn new() -> TanH {
        TanH { }
    }
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

// TODO: make value be the f(x)
    fn derivative(&self, value: f64) -> f64
    {
        (1.0 - (value.tanh() * value.tanh()))
    }
    fn derivativeRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor)
    {
        for i in 0..tensor.size() {
            output[i] = 1.0 - (tensor[i].tanh() * tensor[i].tanh());
        }
    }
    fn derivativeRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor)
    {
        for i in 0..tensor.rows() {
            for j in 0..tensor.cols() {
                output[i][j] = 1.0 - (tensor[i][j].tanh() * tensor[i][j].tanh());
            }
        }
    }

}

pub struct Identity{}
impl Identity {
    pub fn new() -> Identity {
        Identity { }
    }
}
impl IsFunction for Identity {
    fn evaluate(&self, value: f64) -> f64 { value }
    fn evaluateRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor) { output.copy(tensor); }
    fn evaluateRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor) { output.copy(tensor); }

    fn inverse(&self, value: f64) -> f64 { value }
    fn inverseRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor) { output.copy(tensor); }
    fn inverseRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor) { output.copy(tensor); }

    fn derivative(&self, value: f64) -> f64
    {
        1.0
    }
    fn derivativeRank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor)
    {
        output.fill(1.0);
    }
    fn derivativeRank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor)
    {
        output.fill(1.0);
    }
}
