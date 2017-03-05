use Tensor::Vector;
use Tensor::Rank1Tensor;
use rand::Rng;
use rand;
use rand::distributions::{Range, IndependentSample};

use std::ops::{Index, IndexMut};

pub enum Compatibilty {
    Mul,
    Add,
    Sub,
}

#[derive(Clone, Debug)]
pub struct Rank2Tensor {
    m_data: Vector<Rank1Tensor>,
    m_rows: u64,
    m_cols: u64,
}

impl Rank2Tensor {
    /// New Function for creating a Rank2Tensor:
    pub fn new(rows: u64, cols: u64) -> Rank2Tensor
    {
        let mut new_vec = Vector::new();
        for i in 0..rows {
            let mut temp_vec = Rank1Tensor::new(cols);
            new_vec.push(temp_vec);
        }
        Rank2Tensor {
            m_data: new_vec,
            m_rows: rows,
            m_cols: cols,
        }
    }

    pub fn resizeColumns(&mut self, cols: u64)
    {
        println!("Rows: {}", self.m_rows);
        self.m_cols = cols;
        for i in 0..self.m_rows
        {
            self.m_data[i].resize(cols);
        }
    }

    pub fn fillRandom(&mut self)
    {
        let mut rng = rand::thread_rng();
        let range = Range::new(0.0, 1.0);
        for i in 0..self.m_rows {
            for j in 0..self.m_cols {
                self.m_data[i][j] = range.ind_sample(&mut rng);
            }
        }
    }

    pub fn resize(&mut self, rows: u64, cols: u64)
    {
        self.m_data.resize(rows, Rank1Tensor::new(cols));
        for i in 0..rows {
            self.m_data[i].resize(cols);
        }
        self.m_cols = cols;
        self.m_rows = rows;
    }

    pub fn copy(&mut self, other: &Rank2Tensor) {
        self.m_data.resize(other.m_rows, Rank1Tensor::new(other.cols()));
        self.m_rows = other.m_rows;
        self.m_cols = other.m_cols;
        for i in 0..self.m_rows {
            for j in 0..self.m_cols {
                self.m_data[i][j] = other.m_data[i][j];
            }
        }
    }

    pub fn get(&self, row: u64, col: u64) -> &f64 {
        &self.m_data[row][col]
    }

    pub fn set(&mut self, row: u64, col: u64, value: f64) {
        self.m_data[row][col] = value;
    }

    pub fn zeroes(&mut self) {
        for i in 0..self.m_rows {
            for j in 0..self.m_cols {
                self.m_data[i][j] = 0.0;
            }
        }
    }

    pub fn fill(&mut self, value: f64) {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self.m_data[i][j] = value;
            }
        }
    }

    pub fn identity(&mut self) {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                if i == j {
                    self.m_data[i][j] = 1.0;
                } else {
                    self.m_data[i][j] = 0.0;
                }
            }
        }
    }

    pub fn multiplyRank1(&self, other: &Rank1Tensor) -> Rank1Tensor {
        assert!(self.cols() == other.size(), "When multiplying Rank2Tensor x Rank1Tensor the Rank2Tensor's number of columns must be equal to the size of the Rank1Tensor.");

        let mut resultTensor = Rank1Tensor::new(self.rows());

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                resultTensor[i] = resultTensor[i] + (self[i][j] * other[j]);
            }
        }

        resultTensor
    }

    pub fn multiply(&self, other: &Rank2Tensor) -> Rank2Tensor {
        assert!(self.compatible(other, Compatibilty::Mul),
                "Cannot multiply two 2D tensors that aren't compatible.");

        let final_rows = self.rows();
        let final_cols = other.cols();
        let mut resultTensor = Rank2Tensor::new(final_rows, final_cols);

        for i in 0..final_rows {
            for j in 0..final_cols {
                // Compute dot product at this location:
                let mut temp = 0.0;
                for k in 0..self.cols() {
                    temp += self.get(i, k) * other.get(k, j);
                }
                resultTensor.set(i, j, temp);
            }
        }
        resultTensor
    }

    pub fn add(&self, other: &Rank2Tensor) -> Rank2Tensor {
        assert!(self.compatible(other, Compatibilty::Add),
                "Cannot add two 2D Tensors that aren't compatible!");

        let mut resultTensor = Rank2Tensor::new(self.rows(), self.cols());

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                resultTensor[i][j] = self[i][j] + other[i][j];
            }
        }
        resultTensor
    }

    pub fn subSquare(&self, other: &Rank2Tensor) -> Rank2Tensor {
        assert!(self.compatible(other, Compatibilty::Sub),
                "Cannot add two 2D Tensors that aren't compatible!");

        let mut resultTensor = Rank2Tensor::new(self.rows(), self.cols());

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                resultTensor[i][j] = (self[i][j] - other[i][j]) * (self[i][j] - other[i][j]);
            }
        }
        resultTensor
    }

    pub fn sub(&self, other: &Rank2Tensor) -> Rank2Tensor {
        assert!(self.compatible(other, Compatibilty::Sub),
                "Cannot add two 2D Tensors that aren't compatible!");

        let mut resultTensor = Rank2Tensor::new(self.rows(), self.cols());

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                resultTensor[i][j] = self[i][j] - other[i][j];
            }
        }
        resultTensor
    }

    pub fn scale(&mut self, scalar: f64) {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[i][j] = self[i][j] * scalar;
            }
        }
    }

    pub fn rows(&self) -> u64 {
        self.m_rows
    }

    pub fn cols(&self) -> u64 {
        self.m_cols
    }

    pub fn size(&self) -> (u64, u64) {
        (self.m_rows, self.m_cols)
    }

    pub fn countElements(&self) -> u64 {
        self.m_rows * self.m_cols
    }

    pub fn compatible(&self, other: &Rank2Tensor, c: Compatibilty) -> bool {
        let mut returnValue = false;
        match c {
            Compatibilty::Mul => {
                if other.rows() == self.cols() || other.cols() == self.rows() {
                    returnValue = true;
                }
            }
            Compatibilty::Add => {
                if other.rows() == self.rows() && other.cols() == self.cols() {
                    returnValue = true;
                }
            }
            Compatibilty::Sub => {
                if other.rows() == self.rows() && other.cols() == self.cols() {
                    returnValue = true;
                }
            }
        }
        returnValue
    }
    pub fn print(&self) {
        println!("Tensor Rank 2 rows: {} cols: {}", self.rows(), self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                print!("{} ", self.m_data[i][j]);
            }
            println!("");
        }
        println!("");
    }
    /// Test Function for Rank2Tensor:
    pub fn test() -> bool {
        let mut returnValue = true;
        let mut test_tensor = Rank2Tensor::new(5, 5);

        test_tensor.identity();
        for i in 0..test_tensor.rows() {
            for j in 0..test_tensor.cols() {
                if i == j && test_tensor[i][j] != 1.0 {
                    returnValue = false;
                }
            }
        }

        let mut test_tensor2 = Rank2Tensor::new(5, 5);
        test_tensor2.identity();
        test_tensor2.scale(2.0);
        test_tensor.scale(2.0);

        test_tensor2 = test_tensor.multiply(&test_tensor2);
        for i in 0..test_tensor2.rows() {
            for j in 0..test_tensor2.cols() {
                if i == j && test_tensor2[i][j] != 4.0 {
                    returnValue = false;
                }
            }
        }
        returnValue
    }
}

/// Immutable Index
impl Index<u64> for Rank2Tensor {
    type Output = Rank1Tensor;
    fn index<'a>(&'a self, _index: u64) -> &Rank1Tensor {
        &self.m_data[_index]
    }
}
/// Mutable Index
impl IndexMut<u64> for Rank2Tensor {
    fn index_mut<'a>(&'a mut self, _index: u64) -> &'a mut Rank1Tensor {
        &mut self.m_data[_index]
    }
}
