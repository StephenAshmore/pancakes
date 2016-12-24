// use tensor::traits;
use Tensor::Rank1Tensor;
use Tensor::Rank1Tensor::rank1Tensor;
use std::ops::{Add, Mul, Index, IndexMut};

pub struct rank2Tensor {
    m_data: Vec<rank1Tensor>,
    m_rows: u64,
    m_cols: u64,
}

impl rank2Tensor {
    pub fn zeroes(&mut self, rows: u64, cols: u64){
        // self.m_data.resize(rows as usize);
        self.m_rows = rows;
        self.m_cols = cols;
        for i in 0..self.rows() {
            self.m_data.push(Rank1Tensor::new(cols));
            // self.m_data[i as usize].zeroes(cols);
        }
    }

    pub fn fill(&mut self, value: f64) {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[i][j] = value;
            }
        }
    }

    pub fn rows(&self) -> u64 {
        self.m_rows;
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

    // pub fn get(&self, indices: u64) -> f64 {
    //     if indices >= 0 && indices < self.size() {
    //         self.m_data[indices as usize]
    //     }
    //     else {
    //         0.0
    //     }
    // }
    //
    // pub fn set(&mut self, indices: u64, value: f64) {
    //     if indices >= 0 && indices < self.size() {
    //         self.m_data[indices as usize] = value;
    //     }
    // }

    // pub fn multiply(&mut self, scalar: f64) {
    //     for i in 0..self.m_size {
    //         self.m_data[i as usize] *= scalar;
    //     }
    //     // for i in 0..self.data.size() {
    //     //     self.data[i] *= scalar;
    //     // }
    // }

    pub fn print(&self) {
        println!("Tensor Rank 2 rows: {} cols: {}", self.rows(), self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                print!("{} ", self.m_data[i as usize][j as usize]);
            }
            println!("");
        }
        println!("");
    }
}

impl Index<u64> for rank2Tensor {
    type Output = rank1Tensor;
    fn index<'a>(&'a self, _index: u64) -> &rank1Tensor {
        &self.m_data[_index as usize]
    }
}

impl IndexMut<u64> for rank2Tensor {
    fn index_mut<'a>(&'a mut self, _index: u64) -> &'a mut rank1Tensor {
        & mut self.m_data[_index as usize]
    }
}

// impl Mul<rank2Tensor> for rank2Tensor {
//
// }

/// New Function for creating a rank2Tensor:
pub fn new(rows: u64, cols:u64) -> rank2Tensor {
    // println!("Creating new tensor!");
    let mut newVec = Vec::with_capacity(rows as usize, Vec::with_capacity(cols as usize));
    newVec.resize(rows as usize);
    for i in 0..rows {
        newVec[i].resize(cols as usize, 0.0);
    }
    rank2Tensor {
        m_data: newVec,
        m_rows: rows,
        m_cols: cols,
    }
}

// Test Function for rank2Tensor:
// pub fn test() -> bool {
//     let mut returnValue = true;
//     let mut test_tensor = new(5);
//     for i in 0..test_tensor.size() {
//         if test_tensor.get(i) != 0.0 {
//             returnValue = false;
//         }
//     }
//
//     for i in 0..test_tensor.size() {
//         test_tensor.set(i, (i as f64) + 1.0);
//     }
//
//     // scale tensor:
//     test_tensor.multiply(2.0);
//     for i in 0..test_tensor.size() {
//         if test_tensor.get(i) != ((i as f64)+1.0) * 2.0 {
//             returnValue = false;
//         }
//     }
//
//     returnValue
// }
