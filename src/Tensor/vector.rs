use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
pub struct Vector<T> {
    m_data: Vec<T>,
    m_size: u64,
}

impl<T: Clone> Vector<T> {
    pub fn new() -> Vector<T> {
        let mut newVec = Vec::new();
        Vector {
            m_data: newVec,
            m_size: 0,
        }
    }

    pub fn with_capacity(size: u64) -> Vector<T> {
        let mut newVec = Vec::with_capacity(size as usize);
        Vector {
            m_data: newVec,
            m_size: 0,
        }
    }

    pub fn clear(&mut self)
    {
        self.m_data.clear();
        self.m_size = 0;
    }

    pub fn push(&mut self, value: T) {
        self.m_data.push(value);
        self.m_size += 1;
    }

    pub fn push_all(&mut self, other: &Vector<T>) {
        for i in 0..other.m_size {
            self.m_data.push((other.m_data[i as usize]).clone());
        }
        self.m_size = other.m_size;
    }

    pub fn size(&self) -> u64 {
        self.m_size
    }

    pub fn resize(&mut self, size: u64, value: T) {
        assert!(size > 0, "You can't make a negative sized Vector!");

        self.m_data.resize(size as usize, value);
        self.m_size = size;
    }

    /// Build a filled Vector
    pub fn build(size: u64, value: T) -> Vector<T> {
        let mut newVec = Vec::with_capacity(size as usize);
        newVec.resize(size as usize, value);
        Vector {
            m_data: newVec,
            m_size: size,
        }
    }
}

//IMPLEMENTATIONS
/// Inmutable Index operator []
impl<T: Clone> Index<u64> for Vector<T> {
    type Output = T;
    fn index<'a>(&'a self, _index: u64) -> &T {
        &self.m_data[_index as usize]
    }
}
/// Mutable Index operator []
impl<T: Clone> IndexMut<u64> for Vector<T> {
    fn index_mut<'a>(&'a mut self, _index: u64) -> &'a mut T {
        &mut self.m_data[_index as usize]
    }
}

// Clone trait
// impl<T: Copy> Clone for Vector<T> {
//     fn clone(&self) -> Vector<T> {
//         let mut returnVec = Vector::with_capacity(self.m_size);
//         returnVec.push_all(self);
//
//         returnVecf
//     }
// }


pub fn test() -> bool {
    // test here.
    true
}
