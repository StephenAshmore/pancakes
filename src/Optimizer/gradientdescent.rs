use Optimizer::optimizertraits::*;
use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;

pub struct GradientDescent {
    m_learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> GradientDescent {
        GradientDescent {
            m_learning_rate: 0.0001,
        }
    }
}

impl Optimizer for GradientDescent {
    fn optimize(&mut self, input: &Rank2Tensor, output: &mut Rank2Tensor) {

    }

    fn optimize1D(&mut self, input: &Rank1Tensor, output: &mut Rank1Tensor) {

    }
}
