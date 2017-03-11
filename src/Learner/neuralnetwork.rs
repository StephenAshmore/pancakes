use Tensor::Rank2Tensor;
use Tensor::Vector;
use Tensor::Rank1Tensor;
use Learner::Layer;
use Function::functions::*;
use Learner::learnertraits::*;
use Learner::activation_layer::*;
use Optimizer::*;
use Function::functiontraits::*;
use Function::costfunctions::*;

// #[derive(Clone)]
pub struct NeuralNetwork {
    m_blocks: Vec<Vec<Box<Differentiable>>>,
    m_layer_count: u64,
    m_optimizer: Box<Optimizer>,
    m_cost_function: Box<IsCostFunction>,
    m_ready: bool,
    m_inputs: u64,
    m_outputs: u64,
    m_layer_output_counts: Vector<u64>,
}

impl NeuralNetwork {
    pub fn new(optimizer: Box<Optimizer>, cost_function: Box<IsCostFunction>) -> NeuralNetwork
    {
        NeuralNetwork {
            m_blocks: Vec::new(),
            m_layer_count: 0,
            m_optimizer: optimizer,
            m_cost_function: cost_function,
            m_ready: false,
            m_inputs: 0,
            m_outputs: 0,
            m_layer_output_counts: Vector::new(),
        }
    }

    pub fn add(&mut self, block: Box<Differentiable>)
    {
        self.m_layer_count += 1;
        let mut new_vec = Vec::new();
        new_vec.push(block);
        self.m_blocks.push(new_vec);
        self.m_ready = false;
    }

    pub fn concat(&mut self, block: Box<Differentiable>)
    {
        self.m_blocks[(self.m_layer_count-1) as usize].push(block);
        self.m_ready = false;
    }

    pub fn get_block(&mut self, layer: u64, block: u64) -> &mut Box<Differentiable>
    {
        assert!((layer >= 0 && layer < self.m_layer_count), "You can't get a layer that isn't in the neural network.");
        assert!((block >= 0 && (block as usize) < (self.m_blocks[layer as usize].len())), "You can't get a block that isn't in a layer.");

        &mut self.m_blocks[layer as usize][block as usize]
    }

    pub fn concatAtPosition(&mut self, block: Box<Differentiable>, pos: u64)
    {
        assert!(pos < self.m_layer_count, "You can only concatenate blocks at a position less than the number of layers in the neural network.");

        self.m_blocks[(pos) as usize].push(block);
        self.m_ready = false;
    }

    pub fn validate(&mut self, global_inputs: u64)
    {
        // Compute inputs for each layer based on the first inputs (global_inputs)
        // and then subsequent layers based on the total output of the previous layer.
        self.m_inputs = global_inputs;
        let mut current_inputs = global_inputs;
        let mut current_outputs = 0;

        for i in 0..self.m_blocks.len() {
            current_outputs = 0;
            for j in 0..self.m_blocks[i].len() {
                self.m_blocks[i as usize][j as usize].setInputs(current_inputs);
                current_outputs += self.m_blocks[i as usize][j as usize].outputs();
            }
            current_inputs = current_outputs;
            self.m_layer_output_counts.push(current_outputs);
        }
        self.m_outputs = current_outputs;

        self.m_ready = true;
    }

    // pub fn converged() {

    // }

    // TRAIN method for Neural network
    pub fn train(&mut self, features: &Rank2Tensor, labels: &Rank2Tensor)
    {
        if !self.m_ready {
            self.validate(features.cols());
        }
        assert!(self.m_inputs == features.cols(), "Features must have the same number of columns as the number of inputs to the neural network.");
        assert!(self.m_outputs == labels.cols(), "Labels must have the same number of columns as the number of outputs to the neural network.");

        // iterate until fully trained
        // TODO: implement code to stop once convergence ends.
        // while !self.converged() {
        for i in 0..5000
        {
            // println!("Epoch: {}", i);
            // TODO: randomly iterate through training set
            for row in 0..features.rows()
            {
                let mut prediction = Rank1Tensor::new(self.m_outputs);

                self.forward(&features[row], &mut prediction);

                // calculate error of network:
                let mut error = self.m_cost_function.evaluateRank1(&labels[row], &prediction);
                let mut next_error = Rank1Tensor::new(self.inputs());
                self.backprop(&error, &mut next_error); // next_error unused

                for i in 0..self.m_layer_count {
                    for j in 0..self.m_blocks[i as usize].len() {
                        self.m_blocks[i as usize][j as usize].update(&mut self.m_optimizer);
                    }
                }
            }
        }
    }

    pub fn layers(&self) -> u64
    {
        self.m_layer_count
    }

    pub fn output_at_layer_count(&self, layer_number: u64) -> u64
    {
        assert!(self.m_ready, "You cannot get the output at a layer without first validating the neural network!"); // this assert may change to allow this function to call validate.

        assert!((layer_number >= 0 && layer_number < self.m_layer_count), "You cannot get the layer count for a layer that does not exist.");

        self.m_layer_output_counts[layer_number]
    }

    pub fn test() -> bool {
        let mut nn = NeuralNetwork::new(Box::new(GradientDescent::new(Some(0.1))) as Box<Optimizer>, Box::new(difference::new()) as Box<IsCostFunction>);

        // Not the most pleasant syntax for moving a trait object.
        nn.add(Box::new(Layer::new(3)) as Box<Differentiable>);
        nn.add(Box::new(Activation_Layer::new(3, TanH::new())) as Box<Differentiable>);
        nn.add(Box::new(Layer::new(2)) as Box<Differentiable>);
        nn.add(Box::new(Activation_Layer::new(2, TanH::new())) as Box<Differentiable>);
        //nn.concat(Box::new(Layer::new(6, Identity::new())) as Box<Differentiable>);

        // test code:
        let mut layer1_weights = Rank2Tensor::new(3,2);
        layer1_weights[0][0] = 0.1;layer1_weights[0][1] = 0.1;
        layer1_weights[1][0] = 0.0;layer1_weights[1][1] = 0.0;
        layer1_weights[2][0] = 0.1;layer1_weights[2][1] = -0.1;
        let mut layer1_bias = Rank2Tensor::new(1,3);
        layer1_bias[0][0] = 0.1;layer1_bias[0][1] = 0.1;layer1_bias[0][2] = 0.0;

        let mut layer2_weights = Rank2Tensor::new(2, 3);
        layer2_weights[0][0] = 0.1;layer2_weights[0][1] = 0.1;layer2_weights[0][2] = 0.1;
        layer2_weights[1][0] = 0.1;layer2_weights[1][1] = 0.3;layer2_weights[1][2] = -0.1;
        let mut layer2_bias = Rank2Tensor::new(1, 2);
        layer2_bias[0][0] = 0.1;layer2_bias[0][1] = -0.2;

        // set weights of the two layers:
        let mut layer1_vec = Vec::new();
        layer1_vec.push(layer1_weights);layer1_vec.push(layer1_bias);
        let mut layer2_vec = Vec::new();
        layer2_vec.push(layer2_weights);layer2_vec.push(layer2_bias);

        // validate neural network:
        nn.validate(2);

        nn.get_block(0,0).set_weights(&layer1_vec);
        nn.get_block(2,0).set_weights(&layer2_vec);//skip activation layer

        let mut features = Rank2Tensor::new(1, 2);
        features[0][0] = 0.3;
        features[0][1] = -0.2;

        // test feed forward!
        let mut labels = Rank2Tensor::new(1, 2);
        labels[0][0] = 0.1; labels[0][1] = 0.0;

        let mut prediction = Rank1Tensor::new(labels.cols());
        // set the weights of the network:

        nn.forward(&features[0], &mut prediction);
        let mut error = nn.m_cost_function.evaluateRank1(&labels[0], &prediction);
        // println!("Final prediction: {:?}", prediction);
        let mut correct_pred = Rank1Tensor::new(2);
        correct_pred[0] = 0.12525717909304; correct_pred[1] = -0.16268123406035;
        if !prediction.fuzzy_equals(&correct_pred) {
            println!("Failed: Forward did not result in the correct prediction.");
        }

        let mut input_error = Rank1Tensor::new(nn.inputs());
        nn.backprop(&error, &mut input_error);

        let mut layer1_gradient = nn.m_blocks[0][0].first_gradient();
        let mut layer2_gradient = nn.m_blocks[2][0].first_gradient();

        // iterate through all blocks, call update on them.
        for i in 0..nn.layers() {
            for j in 0..nn.m_blocks[i as usize].len() {
                nn.m_blocks[i as usize][j as usize].update(&mut nn.m_optimizer);
            }
        }

        correct_pred[0] = 0.12318119206528; correct_pred[1] = -0.14502768344823;
        nn.forward(&features[0], &mut prediction);
        if !prediction.fuzzy_equals(&correct_pred) {
            println!("Failed: Forward did not result in the correct prediction.");
        }

        // test loading iris:
        let mut iris = Rank2Tensor::new(0,0);
        iris.load_arff("iris.arff".to_string());
        iris.shuffle();

        // TODO: Copy portion failing?
        let mut train_features = iris.copy_portion(0, 0, iris.rows() * 2 / 3, 4);
        let mut train_labels = iris.copy_portion(0, 4, iris.rows() * 2 / 3, 1);
        let mut test_features = iris.copy_portion(iris.rows() * 2 /3, 0, iris.rows() * 1 / 3, 4);
        let mut test_labels = iris.copy_portion(iris.rows() * 2 /3, 4, iris.rows() * 1 / 3, 1);
        println!("Train features rows,cols: {},{}", train_features.rows(), train_features.cols());
        println!("Train labels rows,cols: {},{}", train_labels.rows(), train_labels.cols());
        println!("Test features rows,cols: {},{}", test_features.rows(), test_features.cols());
        println!("Test labels rows,cols: {},{}", test_labels.rows(), test_labels.cols());

        // Test on iris dataset:
        let mut nn2 = NeuralNetwork::new(Box::new(GradientDescent::new(Some(0.1))) as Box<Optimizer>, Box::new(difference::new()) as Box<IsCostFunction>);
        nn2.add(Box::new(Layer::new(16)) as Box<Differentiable>);
        nn2.add(Box::new(Activation_Layer::new(16, TanH::new())) as Box<Differentiable>);
        nn2.add(Box::new(Layer::new(8)) as Box<Differentiable>);
        nn2.add(Box::new(Activation_Layer::new(8, TanH::new())) as Box<Differentiable>);
        nn2.add(Box::new(Layer::new(1)) as Box<Differentiable>);

        nn2.train(&train_features, &train_labels);

        // validate on test set
        let mut test_prediction = Rank1Tensor::new(test_labels.cols());
        println!("Test set size: {}", test_features.rows());
        let sse_func = SSE::new();
        let mut mis = 0;
        for i in 0..test_features.rows() {
            nn2.forward(&test_features[i], &mut test_prediction);
            test_prediction[0] = test_prediction[0].round();
            if test_prediction[0] != test_labels[i][0] {
                mis += 1;
            }
            // error += temp_error[0];
            // println!("Prediction: {} versus target: {}", test_prediction[0], test_labels[i][0]);
        }
        println!("Misclassifications: {}", mis);

        true
    }
}

impl Differentiable for NeuralNetwork {
    fn inputs(&self) -> u64
    {
        self.m_inputs
    }

    fn outputs(&self) -> u64
    {
        self.m_outputs
    }

    fn setInputs(&mut self, new_inputs: u64)
    {
        self.m_inputs = new_inputs;
        self.validate(new_inputs);
    }

    fn set_weights(&mut self, weights: &Vec<Rank2Tensor>)
    {
        // This one will be a bit more complicated to handle.
        // For now, do nothing.
    }

    fn first_gradient(&self) -> Rank1Tensor
    {
        assert!(self.m_ready, "The neural network must be validated before calling the first_gradient.");
        self.m_blocks[0][0].first_gradient()
    }

    fn forward(&mut self, input: &Rank1Tensor, prediction: &mut Rank1Tensor)
    {
        if !self.m_ready {
            self.validate(input.size());
        }
        assert!(prediction.size() == self.m_outputs,
            "The prediction Rank1Tensor must be the same size as the number of neurons in this layer");
        assert!(self.m_inputs != 0 && input.size() == self.m_inputs,
            "The input Rank1Tensor must be the same size as the number of inputs in this layer!");


        // actually go through the layers and stuff:
        // give the correct input to the next layer:
        let mut current_input = Rank1Tensor::new(input.size());
        current_input.copy(input);
        for i in 0..self.m_layer_count {
            let mut current_output = Rank1Tensor::new(self.output_at_layer_count(i));

            // forward into each block with the current input:
            let mut start_position = 0;
            for j in 0..self.m_blocks[i as usize].len() {
                // forward input here:
                let block_output_count = self.m_blocks[i as usize][j as usize].outputs();
                let mut block_output = Rank1Tensor::new(block_output_count);
                self.m_blocks[i as usize][j as usize].forward(&current_input, &mut block_output);
                current_output.slice_from(&block_output, start_position);
                start_position += block_output_count;
            }
            current_input.copy(&current_output);
        }
        prediction.copy(&current_input);
    }

    fn backprop(&mut self, previous_error: &Rank1Tensor, error: &mut Rank1Tensor)
    {
        //TODO: This method is probably very, very slow due to copying error vectors.
        // This will be slightly mitigated in mini-batches, but not entirely removed.
        // Perhaps we should look into using references to the specific error vectors?
        assert!(self.m_ready == true,
            "You cannot call backprop on a neural network that has not been validated first.");

        assert!(previous_error.size() == self.m_outputs,
            "The previous error passed to a neural network must have the same size as the number of outputs in the neural network.");

        // actually go through the layers and stuff:
        // give the correct input to the next layer:
        let mut current_error = Rank1Tensor::new(previous_error.size());
        current_error.copy(previous_error);


        for i in (0..self.m_layer_count).rev() {
            let mut error_start_position = 0; let mut next_error_start_position = 0;
            let mut total_layer_inputs = 0;
            // calculate total inputs to this layer:
            for k in 0..self.m_blocks[i as usize].len() {
                total_layer_inputs += self.m_blocks[i as usize][k as usize].inputs();
            }

            let mut next_block_error = Rank1Tensor::new(total_layer_inputs);
            for j in 0..self.m_blocks[i as usize].len() {
                let mut single_block_error = Rank1Tensor::new(self.m_blocks[i as usize][j as usize].outputs());
                single_block_error.copy_slice(&current_error, error_start_position, None);
                let mut next_single_block_error = Rank1Tensor::new(self.m_blocks[i as usize][j as usize].inputs());

                // give current error to this block and store its backpropagated error
                self.m_blocks[i as usize][j as usize].backprop(&single_block_error, &mut next_single_block_error);

                next_block_error.slice_from(&next_single_block_error, next_error_start_position);
                next_error_start_position += next_single_block_error.size();

                error_start_position += self.m_blocks[i as usize][j as usize].outputs();
            }
            current_error.copy(&next_block_error);
        }

        error.copy(&current_error);
    }

    fn update(&mut self, optimizer: &mut Box<Optimizer>)
    {
        // iterate through all blocks, call update on them.
        for i in 0..self.layers() {
            for j in 0..self.m_blocks[i as usize].len() {
                self.m_blocks[i as usize][j as usize].update(optimizer);
            }
        }
    }


    fn forwardBatch(&mut self, features: Rank2Tensor) {

    }

    fn backpropBatch(&mut self) {

    }
}
