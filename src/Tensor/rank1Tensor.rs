// use tensor::traits;
// pub mod Rank1Tensor;
use std::ops::{Add, Mul, Index, IndexMut};
use Tensor::Vector;

#[derive(Clone, Debug)]
pub struct Rank1Tensor {
    m_data: Vector<f64>,
    m_size: u64,
}

impl Rank1Tensor {
    /// New Function for creating a Rank1Tensor:
    pub fn new(s: u64) -> Rank1Tensor {
        Rank1Tensor {
            m_data: Vector::build(s, 0.0),
            m_size: s,
        }
    }

    pub fn copy(&mut self, other: &Rank1Tensor) {
        self.m_data.resize(other.m_size, 0.0);
        self.m_size = other.m_size;
        for i in 0..self.m_size {
            self.m_data[i] = other.m_data[i];
        }
    }

    pub fn copy_slice(&mut self, other: &Rank1Tensor, start_position: u64) {
        assert!(other.size() < self.m_size - start_position, "You cannot copy a Rank1Tensor into a slice of another Rank1Tensor if the slice is not big enough!");

        for i in 0..other.size() {
            self.m_data[i + start_position] = other[i];
        }
    }

    pub fn zeroes(&mut self, s: u64) {
        self.m_data.resize(s, 0.0);
        self.m_size = s;
    }

    pub fn fill(&mut self, value: f64) {
        for i in 0..self.size() {
            self[i] = value;
        }
    }

    pub fn size(&self) -> u64 {
        self.m_size
    }

    pub fn get(&self, indices: u64) -> f64 {
        if indices >= 0 && indices < self.size() {
            self.m_data[indices]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, indices: u64, value: f64) {
        if indices >= 0 && indices < self.size() {
            self.m_data[indices] = value;
        }
    }

    pub fn add(&self, other: &Rank1Tensor) -> Rank1Tensor {
        let mut result_tensor = Rank1Tensor::new(self.m_size);
        for i in 0..self.m_size {
            result_tensor[i] = self[i] + other[i];
        }

        result_tensor
    }

    pub fn scale(&mut self, scalar: f64) {
        for i in 0..self.m_size {
            self.m_data[i] *= scalar;
        }
        // for i in 0..self.data.size() {
        //     self.data[i] *= scalar;
        // }
    }

    pub fn multiply(&self, other: &Rank1Tensor) {}

    pub fn print(&self) {
        println!("Tensor size: {}", self.m_size);
        for i in 0..self.m_size {
            print!("{} ", self.m_data[i]);
        }
        println!("");
    }

    /// Test Function for Rank1Tensor:
    pub fn test() -> bool {
        let mut returnValue = true;
        let mut test_tensor = Rank1Tensor::new(5);
        for i in 0..test_tensor.size() {
            if test_tensor.get(i) != 0.0 {
                returnValue = false;
            }
        }

        for i in 0..test_tensor.size() {
            test_tensor.set(i, (i as f64) + 1.0);
        }

        // scale tensor:
        test_tensor.scale(2.0);
        for i in 0..test_tensor.size() {
            if test_tensor.get(i) != ((i as f64) + 1.0) * 2.0 {
                returnValue = false;
            }
        }

        returnValue
    }
}

//IMPLEMENTATIONS
/// Inmutable Index operator []
impl Index<u64> for Rank1Tensor {
    type Output = f64;
    fn index<'a>(&'a self, _index: u64) -> &f64 {
        &self.m_data[_index]
    }
}
/// Mutable Index operator []
impl IndexMut<u64> for Rank1Tensor {
    fn index_mut<'a>(&'a mut self, _index: u64) -> &'a mut f64 {
        &mut self.m_data[_index]
    }
}
// Clone trait
// impl Clone for Rank1Tensor {
//     fn clone(&self) -> Rank1Tensor {
//         *self
//     }
// }

// impl Mul<Rank1Tensor> for Rank1Tensor {
//
// }
