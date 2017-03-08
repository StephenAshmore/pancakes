use Optimizer::optimizertraits::*;
use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Function::functiontraits::*;

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
    // what should input be?
    // GRADIENT IS BLAME
    fn optimize(&mut self, weights: &mut Rank2Tensor, bias: &mut Rank1Tensor, input: &mut Rank1Tensor, gradient: &Rank1Tensor) {
        // input size should be equal to the number of columns in weights.
        // assert!(weights.cols() == input.size(), "The number of inputs must equal the columns of the weights for gradient descent to make sense.");

        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                // what should I do here?
                // weights[i][j] -= self.m_learning_rate * blame[i];
                weights[i][j] += self.m_learning_rate * gradient[i] * input[j];
            }
            bias[i] += self.m_learning_rate * gradient[i];
        }
    }
}
