// use tensor::traits;
// pub mod rank1Tensor;
use std::ops::{Add, Mul, Index, IndexMut};

pub struct rank1Tensor {
    m_data: Vec<f64>,
    m_size: u64,
}

impl rank1Tensor {
    pub fn zeroes(&mut self, s: u64){
        self.m_data.resize(s as usize, 0.0);
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
            self.m_data[indices as usize]
        }
        else {
            0.0
        }
    }

    pub fn set(&mut self, indices: u64, value: f64) {
        if indices >= 0 && indices < self.size() {
            self.m_data[indices as usize] = value;
        }
    }

    pub fn multiply(&mut self, scalar: f64) {
        for i in 0..self.m_size {
            self.m_data[i as usize] *= scalar;
        }
        // for i in 0..self.data.size() {
        //     self.data[i] *= scalar;
        // }
    }

    pub fn print(&self) {
        println!("Tensor size: {}", self.m_size);
        for i in 0..self.m_size {
            print!("{} ", self.m_data[i as usize]);
        }
        println!("");
    }
}

//IMPLEMENTATIONS
/// Inmutable Index operator []
impl Index<u64> for rank1Tensor {
    type Output = f64;
    fn index<'a>(&'a self, _index: u64) -> &f64 {
        &self.m_data[_index as usize]
    }
}
/// Mutable Index operator []
impl IndexMut<u64> for rank1Tensor {
    fn index_mut<'a>(&'a mut self, _index: u64) -> &'a mut f64 {
        & mut self.m_data[_index as usize]
    }
}
/// Clone trait
impl Clone for rank1Tensor {
    fn clone(&self) -> rank1Tensor {
        *self
    }
}

// impl Mul<rank1Tensor> for rank1Tensor {
//
// }

/// New Function for creating a rank1Tensor:
pub fn new(s: u64) -> rank1Tensor {
    // println!("Creating new tensor!");
    let mut newVec = Vec::with_capacity(s as usize);
    newVec.resize(s as usize, 0.0);
    rank1Tensor {
        m_data: newVec,
        m_size: s
    }
}

/// Test Function for rank1Tensor:
pub fn test() -> bool {
    let mut returnValue = true;
    let mut test_tensor = new(5);
    for i in 0..test_tensor.size() {
        if test_tensor.get(i) != 0.0 {
            returnValue = false;
        }
    }

    for i in 0..test_tensor.size() {
        test_tensor.set(i, (i as f64) + 1.0);
    }

    // scale tensor:
    test_tensor.multiply(2.0);
    for i in 0..test_tensor.size() {
        if test_tensor.get(i) != ((i as f64)+1.0) * 2.0 {
            returnValue = false;
        }
    }

    returnValue
}
