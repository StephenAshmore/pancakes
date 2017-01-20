use Tensor::Rank2Tensor;
use Tensor::Rank1Tensor;
use Tensor::Vector;

pub struct NeuralBlock {
    m_weights: Vector<Rank2Tensor>,
    m_bias: Vector<Rank1Tensor>,
}

impl NeuralBlock {
    

    pub fn test() -> bool {

        true
    }
}
