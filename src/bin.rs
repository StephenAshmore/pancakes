extern crate pancakes;

use pancakes::test;
use pancakes::Tensor::*;

pub fn main() {
  test();
  let mut tensor1 = Rank2Tensor::new(10, 10);
  tensor1.identity();
  tensor1.print();
}
