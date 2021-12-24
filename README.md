


# MLP MNIST-fashion with PyTorch
MLP implementation in Python with PyTorch for the MNIST-fashion dataset (90+ on test).

1. [General](#General)
    - [Background](#background)
    - [Models Structure](https://github.com/tomershay100/MLP-MNIST-fashion-with-PyTorch/blob/main/README.md#models-structure)
    - [Running Instructions](https://github.com/tomershay100/MLP-MNIST-fashion-with-PyTorch/blob/main/README.md#running-instructions)
2. [Dependencies](#dependencies) 
3. [Installation](#installation)

## General

### Background
Implementation of a neural network on the MNIST dataset, which takes as an input a ``28*28`` grayscale image (``784`` floating point values of pixels between ``0-255``).

### Models Structure
The program contains about seven models of different networks, implemented through ``pytorch``. The last layer size of all the networks is ``10 neurons`` with the ``Softmax`` activation function.

During learning, the network verifies its accuracy on an independent set of data on which learning is not performed. This group is called a ``validation set``. After all the ``epochs``, the network saves its best state, the weights that resulted the maximum accuracy on the validation set, to prevent overfitting.

Finally, the network exports a graph of the accuracy on the training, validation, and the testing sets, by the number of epochs, and prints the final accuracy on the testing set.

The seven models architecture:
1. ``Model A``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** SGD
	* **learning rate:** 0.12
	* No batch normalization, no dropout
***
2. ``Model B``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.0001
	* No batch normalization, no dropout
***
3. ``Model C``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.0001
	* **dropout:** 20% on the 3rd layer (size of 50)
	* No batch normalization
***
4. ``Model D``:
	* **number of hidden layers:** 2
	* **sizes of the layers:** [786, 100, 50, 10]
	* **activation function:** [ReLU, ReLU, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.01
	* **batch normalization:** before the activation function (ReLU) on each of the hidden layers.
	* No dropout
***
5. ``Model E``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 128, 64, 10, 10, 10]
	* **activation function:** [ReLU, ReLU, ReLU, ReLU, Softmax]
	* **optimizer:** SGD
	* **learning rate:** 0.1
	* No batch normalization, no dropout
***
6. ``Model F``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 128, 64, 10, 10, 10]
	* **activation function:** [Sigmoid, Sigmoid, Sigmoid, Sigmoid, Softmax]
	* **optimizer:** ADAM
	* **learning rate:** 0.001
	* No batch normalization, no dropout
***
7. ``Best Model``:
	* **number of hidden layers:** 4
	* **sizes of the layers:** [786, 512, 256, 128, 64, 10]
	* **activation function:** [Leaky ReLU, Leaky ReLU, Leaky ReLU, Leaky ReLU, Softmax]
	* **optimizer:** starts with ADAM, then to SGD
	* **learning rate:** 0.001
	* **batch normalization:** before the activation function (Leaky ReLU) on each of the hidden layers.
	* **dropout:** 10% on the input layer (size of 784), 3rd layer (size of 256), and 5th layer (size of 64).

To get the best percentages on the testing set (90+), our experiments showed that the ``Best Model`` should be run for about ``30`` epochs, with ``batch size = 64`` and ``validation percentage = 10%``.

### Running Instructions

The program gets several arguments, and this can be seen with the ``-h`` flag when running. A total of about seven arguments can be sent:
* flag ```-train_x TRAIN_X_PATH``` for the training images file path (file that contains 784 values in each row).
* flag ```-train_y TRAIN_Y_PATH``` for the training labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file).
* flag ```-test_x TEST_X_PATH``` for the testing images file path (file that contains 784 values in each row).
* flag ```-test_y TEST_Y_PATH``` for the testing labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file).
* flag ```-lr LEARNING_RATE``` for the learning rate that will be used while training the model (``default value = 0.1``).
* flag ```-e EPOCHS``` for the number of epochs (``default value = 20``).
* flag ```-size HIDDEN_LAYER_SIZE``` for the hidden layer size (``default value = 128``).


running example:
```
	$ python3 neural_net.py -train_x train_x -train_y train_y -test_x test_x -test_y test_y -lr 0.03
```

Note that for using the dataset given in this repo, you need to unzip the dataset.zip folder (for example, using 7-zip)
## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* Git
* [NumPy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [Argparse](https://pypi.org/project/argparse/)
* 

## Installation

1. Open the terminal.
2. Clone the project by:
	```
	$ git clone https://github.com/tomershay100/MLP-MNIST-fashion-with-PyTorch.git
	```	
3. Enter the project folder:
	```
	$ cd MLP-MNIST-fashion-with-PyTorch
	```
4. You can unzip the ``dataset.zip`` for local running:
	```
	$ unzip dataset.zip
	```
5. Run the ```main.py``` file with your favorite parameters:
	```
	$ python3 main.py -e 30 -validate 10 -model BestModel -batch_size 64
	 ```
