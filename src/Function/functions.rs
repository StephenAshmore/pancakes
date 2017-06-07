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
    fn evaluate_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor) {
        for i in 0..tensor.size() {
            output[i] = tensor[i].tanh();
        }
    }
    fn evaluate_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor){
        for i in 0..tensor.rows() {
            for j in 0..tensor.cols() {
                output[i][j] = tensor[i][j].tanh();
            }
        }
    }

    fn inverse(&self, value: f64) -> f64{
        value.atanh()
    }
    fn inverse_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor){
        for i in 0..tensor.size() {
            output[i] = tensor[i].atanh();
        }
    }
    fn inverse_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor){
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
    fn derivative_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor)
    {
        for i in 0..tensor.size() {
            output[i] = 1.0 - (tensor[i].tanh() * tensor[i].tanh());
        }
    }
    fn derivative_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor)
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
    fn evaluate_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor) { output.copy(tensor); }
    fn evaluate_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor) { output.copy(tensor); }

    fn inverse(&self, value: f64) -> f64 { value }
    fn inverse_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor) { output.copy(tensor); }
    fn inverse_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor) { output.copy(tensor); }

    fn derivative(&self, value: f64) -> f64
    {
        1.0
    }
    fn derivative_rank1(&self, tensor: &Rank1Tensor, output: &mut Rank1Tensor)
    {
        output.fill(1.0);
    }
    fn derivative_rank2(&self, tensor: &Rank2Tensor, output: &mut Rank2Tensor)
    {
        output.fill(1.0);
    }
}
