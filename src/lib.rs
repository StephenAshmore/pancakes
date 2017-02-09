extern crate rand;

mod Tensor;
mod Learner;
mod Function;
mod Optimizer;

pub fn test() {
    println!("Testing Pancakes, the Neural Network Library built for Rust!");

    print!("Tensor Test: ");
    Tensor::Test();
    println!("Passed");


    print!("Learner Test: ");
    Learner::Test();
    println!("Passed");

}
