use Tensor::Vector;
use Tensor::Rank1Tensor;
use rand;
use rand::distributions::{Range, IndependentSample};
use std::io::prelude::*;
use std::fs::File;
use std::error::Error;

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

    pub fn load(&mut self, dataset: String)
    {
        self.m_data.clear();
        self.m_rows = 0; self.m_cols = 0;
        let mut file = match File::open(&dataset) {
            // The `description` method of `io::Error` returns a string that
            // describes the error
            Err(why) => panic!("couldn't open {}: {}", dataset,
                                                       why.description()),
            Ok(file) => file,
        };

        // Read the file contents into a string, returns `io::Result<usize>`
        let mut s = String::new();
        match file.read_to_string(&mut s) {
            Err(why) => panic!("couldn't read {}: {}", dataset,
                                                       why.description()),
            Ok(_) => (),
        }

        let mut column_counter = 0;
        let mut row_counter = 0;
        let mut iter = s.lines();
        let mut class_names: Vec<String> = Vec::new();

        for line in iter {
            let mut lower = &line.to_lowercase();
            if !lower.starts_with("%") {
                if lower.find("@data") != None {
                    // set up to start doing lines:
                    self.m_cols = column_counter;
                    println!("Columns: {}", self.m_cols);
                }
                else if lower.find("@attribute") != None {
                    column_counter += 1;
                    if lower.find("class") != None {
                    let mut iter3 = lower.split_whitespace();
                        let mut temp_count = 0;
                        for j in iter3 {
                            if temp_count == 2 {
                                // loop through this bloody list
                                let mut temp_s = j.clone().trim_matches('{');
                                temp_s = temp_s.trim_matches('}');
                                let mut iter4 = temp_s.split(',');
                                for k in iter4 {
                                    class_names.push(k.to_string());
                                }
                            }
                            temp_count += 1;
                        }
                    }
                }
                else if lower.find("@") == None && !lower.is_empty() {
                    // split up based on commas
                    // create a new Rank1Tensor based just on this row
                    // println!("Column Counter: {}", column_counter);
                    let mut row = Rank1Tensor::new(column_counter);
                    let mut iter2 = lower.split(",");
                    let mut temp_counter = 0;
                    for i in iter2 {
                        let mut res = i.to_string().parse::<f64>();
                        if res.is_ok() {
                            row[temp_counter] = res.unwrap();
                        }
                        else {
                            for j in 0..class_names.len() {
                                if class_names[j] == i.to_string() {
                                    row[temp_counter] = j as f64;
                                }
                            }
                        }
                        temp_counter += 1;
                    }
                    // push that rank1tensor back
                    self.m_data.push(row);
                    row_counter += 1;
                }
            }
        }

        self.m_rows = row_counter;
        println!("Test arff read: {:?}", self.m_data);
    }

    pub fn resize_columns(&mut self, cols: u64)
    {
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

    pub fn copy(&mut self, other: &Rank2Tensor)
    {
        self.m_data.resize(other.m_rows, Rank1Tensor::new(other.cols()));
        self.m_rows = other.m_rows;
        self.m_cols = other.m_cols;
        for i in 0..self.m_rows {
            for j in 0..self.m_cols {
                self.m_data[i][j] = other.m_data[i][j];
            }
        }
    }

    pub fn get(&self, row: u64, col: u64) -> &f64
    {
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

    pub fn multiply_rank1(&self, other: &Rank1Tensor) -> Rank1Tensor
    {
        assert!(self.cols() == other.size(), "To multiply by a rank1tensor, the number of columns and the size of the tensor must be the same.");

        let mut resultTensor = Rank1Tensor::new(self.rows());

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                resultTensor[i] = resultTensor[i] + (self[i][j] * other[j]);
            }
        }

        resultTensor
    }

    pub fn multiply_rank1_transpose(&self, other: &Rank1Tensor) -> Rank1Tensor
    {
        assert!(self.rows() == other.size(), "To invert and multiply by a rank1tensor, the number of columns and the size of the tensor must be the same.");
        let mut result_tensor = Rank1Tensor::new(self.cols());

        for j in 0..self.cols() {
            for i in 0..self.rows() {
                result_tensor[j] = result_tensor[j] + (self[i][j] * other[i]);
            }
        }

        result_tensor
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
        // test load:
        let mut test_load = Rank2Tensor::new(0,0);
        test_load.load(String::from("iris.arff"));

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
