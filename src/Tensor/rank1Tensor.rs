// use tensor::traits;
// pub mod Rank1Tensor;
use std::cmp;
use std::ops::{Index, IndexMut};
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

    pub fn resize(&mut self, size: u64) {
        self.m_data.resize(size, 0.0);
    }

    pub fn copy(&mut self, other: &Rank1Tensor) {
        self.m_data.resize(other.m_size, 0.0);
        self.m_size = other.m_size;
        for i in 0..self.m_size {
            self.m_data[i] = other.m_data[i];
        }
    }

    pub fn equals(&mut self, other: &Rank1Tensor) -> bool
    {
        if self.size() != other.size() {
            return false;
        }

        for i in 0..self.m_size {
            if self.m_data[i] != other.m_data[i] {
                return false;
            }
        }

        true
    }

    pub fn fuzzy_equals(&mut self, other: &Rank1Tensor) -> bool
    {
        if self.size() != other.size() {
            return false;
        }

        for i in 0..self.m_size {
            let diff = self.m_data[i] - other.m_data[i];
            if diff * diff > 0.0001 {
                return false;
            }
        }

        true
    }

    pub fn slice_from(&mut self, other: &Rank1Tensor, start_position: u64)
    {
        assert!(other.size() <= self.m_size - start_position, "You cannot copy a Rank1Tensor into a slice of another Rank1Tensor if the slice is not big enough!");

        for i in 0..other.size() {
            self.m_data[i + start_position] = other[i];
        }
    }

    pub fn copy_slice(&mut self, other: &Rank1Tensor, start_position: u64, end_position: Option<u64>)
    {
        assert!(other.size() > start_position, "You can't start a position that is outside the bounds of the other tensor.");
        let target_size: u64;
        if end_position.is_none() {
            target_size = cmp::min(self.m_size, other.size() - start_position);
        }
        else {
            target_size = end_position.unwrap() - start_position;
        }

        assert!(target_size <= self.size(),
            "You cannot copy a slice of a Rank1Tensor into another Rank1Tensor if the destination is not big enough!");

        for i in 0..target_size {
            self.m_data[i] = other[i + start_position];
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
        if indices < self.size() {
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

    pub fn sub_square(&self, other: &Rank1Tensor) -> Rank1Tensor {
        let mut result_tensor = Rank1Tensor::new(self.m_size);
        for i in 0..self.m_size {
            result_tensor[i] = (self[i] - other[i]) * (self[i] - other[i]);
        }

        result_tensor
    }

    pub fn sub(&self, other: &Rank1Tensor) -> Rank1Tensor {
        let mut result_tensor = Rank1Tensor::new(self.m_size);
        for i in 0..self.m_size {
            result_tensor[i] = self[i] - other[i];
        }

        result_tensor
    }

    pub fn scale(&mut self, scalar: f64) {
        for i in 0..self.m_size {
            self.m_data[i] *= scalar;
        }
    }

    pub fn multiply(&self, other: &Rank1Tensor, result: &mut Rank1Tensor) {
        assert!(self.size() == other.size(), "To multiply two Rank1Tensors they should be the same size.");
        if result.size() != self.size() {
            result.resize(self.size());
        }
        for i in 0..self.size() {
            result[i] = self.m_data[i] * other[i];
        }
    }

    pub fn print(&self) {
        println!("Tensor size: {}", self.m_size);
        for i in 0..self.m_size {
            print!("{} ", self.m_data[i]);
        }
        println!("");
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

#[cfg(test)]
mod tests {
    use super::Rank1Tensor;

    #[test]
    pub fn rank1_tensor_new() {
        let mut test_tensor = Rank1Tensor::new(5);
        for i in 0..test_tensor.size() {
            assert!(test_tensor[i] == 0.0, "Rank 1 Tensor new method is broken.");
        }
    }

    #[test]
    pub fn rank1_tensor_scale() {
        let mut test_tensor = Rank1Tensor::new(5);
        for i in 0..test_tensor.size() {
            test_tensor.set(i, (i as f64) + 1.0);
        }

        // scale tensor:
        test_tensor.scale(2.0);
        for i in 0..test_tensor.size() {
            assert!(test_tensor.get(i) ==  ((i as f64) + 1.0) * 2.0,
                    "Rank 1 Tensor scale broken.");
        }
    }
}