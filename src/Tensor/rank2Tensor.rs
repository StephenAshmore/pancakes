// use tensor::traits;
use Tensor::Rank1Tensor;
use Tensor::Vector;

use std::ops::{Add, Mul, Index, IndexMut};

pub enum Compatibilty {
    Mul,
    Add,
    Sub,
}

pub struct Rank2Tensor {
    m_data: Vector<Vector<f64>>,
    m_rows: u64,
    m_cols: u64,
}

impl Rank2Tensor {
    /// New Function for creating a Rank2Tensor:
    pub fn new(rows: u64, cols:u64) -> Rank2Tensor {
        let mut newVec = Vector::new();
        for i in 0..rows {
            newVec.push(Vector::build(cols, 0.0));
        }
        Rank2Tensor {
            m_data: newVec,
            m_rows: rows,
            m_cols: cols,
        }
    }

    pub fn zeroes(&mut self){
        for i in 0..self.m_rows {
            for j in 0..self.m_cols {
                self[i][j] = 0.0;
            }
        }
    }

    pub fn fill(&mut self, value: f64) {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[i][j] = value;
            }
        }
    }

    pub fn identity(&mut self) {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                if i == j {
                    self[i][j] = 1.0;
                }
                else {
                    self[i][j] = 0.0;
                }
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

    pub fn compatible(self, other: Rank2Tensor, c: Compatibilty) -> bool {
        let mut returnValue = false;
        match c {
            Compatibilty::Mul => {
                if other.rows() == self.cols() || other.cols() == self.rows() {
                    returnValue = true;
                }
            },
            Compatibilty::Add => {
                if other.rows() == self.rows() && other.cols() == self.cols() {
                    returnValue = true;
                }
            },
            Compatibilty::Sub => {
                if other.rows() == self.rows() && other.cols() == self.cols() {
                    returnValue = true;
                }
            },
        }
        returnValue
    }


    pub fn print(&self) {
        println!("Tensor Rank 2 rows: {} cols: {}", self.rows(), self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                print!("{} ", (self[i])[j]);
            }
            println!("");
        }
        println!("");
    }
}

impl Index<u64> for Rank2Tensor {
    type Output = Vector<f64>;
    fn index<'a>(&'a self, _index: u64) -> &Vector<f64>
    {
        &self.m_data[_index]
    }
}

impl IndexMut<u64> for Rank2Tensor {
    fn index_mut<'a>(&'a mut self, _index: u64) -> &'a mut Vector<f64> {
        & mut self.m_data[_index]
    }
}

// impl Copy for Rank2Tensor {
//
// }

impl Clone for Rank2Tensor {
    fn clone(&self) -> Rank2Tensor {
        *self
    }
}

impl Mul for Rank2Tensor {
    type Output = Rank2Tensor;
    fn mul(self, other: Rank2Tensor) -> Rank2Tensor
    {
        assert!(self.compatible(other, Compatibilty::Mul),"Cannot multiply two 2D tensors that aren't compatible.");

        let final_rows = self.rows();
        let final_cols = other.cols();
        let mut resultTensor = Rank2Tensor::new(final_rows, final_cols);

        for i in 0..final_rows {
            for j in 0..final_cols {
                // Compute dot product at this location:
                for k in 0..self.cols() {
                    resultTensor[i][j] += self[i][k] * other[k][j];
                }
            }
        }
        resultTensor
    }
}

// Test Function for Rank2Tensor:
pub fn test() -> bool {
    let mut returnValue = true;
    let mut test_tensor = Rank2Tensor::new(5, 5);

    test_tensor.identity();
    test_tensor.print();

    returnValue
}
