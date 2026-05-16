# MNIST Digit Classifier — NumPy from Scratch

A handwritten digit classifier trained on the MNIST dataset, built using **only NumPy** — no PyTorch, no TensorFlow, no ML frameworks of any kind. Every component including forward propagation, backpropagation, and gradient descent is implemented manually.

Built to develop ground-up understanding of how modern deep learning frameworks work under the hood.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 92% |
| Dataset Size | 19,999 samples |
| Training Set | 18,999 samples |
| Dev/Test Set | 1,000 samples |

## Architecture
Input Layer     →   784 neurons  (28×28 flattened pixels)
Hidden Layer    →   10 neurons   (ReLU activation)
Output Layer    →   10 neurons   (Sigmoid activation) Then Softmax

## Implementation Details

Everything is implemented from scratch in NumPy:

- **Forward propagation** — manual matrix multiplication, ReLU and Sigmoid activations
- **Backpropagation** — hand-derived gradients for each layer
- **Optimizer** — plain SGD with learning rate 0.01
- **Weight init** — random initialization centered around zero
- **Data normalization** — pixel values divided by 255

## Training Config

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 18,999 (full batch) |
| Learning Rate | 0.01 |
| Optimizer | SGD |

## Why No Frameworks?

Using only NumPy forces you to implement every operation that PyTorch or TensorFlow abstracts away — matrix multiplications, activation derivatives, chain rule through layers. This makes the math explicit and builds real intuition for what happens inside a neural network.

## Requirements

```bash
pip install numpy pandas matplotlib
```

## Run

Open `mnist_classifier.ipynb` in Jupyter and run all cells.
