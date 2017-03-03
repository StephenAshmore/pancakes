use Optimizer::optimizertraits::*;
use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Function::functiontraits::IsFunction;

// equation: (BLAME) * a`(w*x_i)x_i

pub struct GradientDescent {
    m_learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: Option<f64>) -> GradientDescent {
        GradientDescent {
            m_learning_rate: learning_rate.unwrap_or(0.00001),

        }
    }
}

// Pass in net_input to this
impl Optimizer for GradientDescent {
    fn optimize(&mut self, weights: &mut Rank2Tensor, input: &Rank1Tensor, blame: &Rank1Tensor, activation: Box<IsFunction>) {
        // input size should be equal to the number of columns in weights.
        assert!(weights.cols() == input.size(), "The number of inputs must equal the columns of the weights for gradient descent to make sense.");

        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                weights[i][j] -= self.m_learning_rate * ( blame[i] * (activation.derivative( weights[i][j] * input[j] )) * input[j]);
            }
        }
    }
}
