# Neural Network with Backpropagation in C

A comprehensive implementation of a feedforward neural network with backpropagation algorithm written in C. This project demonstrates fundamental machine learning concepts and includes multiple activation functions, momentum-based optimization, and various logic gate problems.

## Features

- **Multiple Activation Functions**: Sigmoid, Tanh, ReLU, Leaky ReLU
- **Advanced Optimization**: Momentum-based gradient descent
- **Smart Initialization**: Xavier weight initialization
- **Early Stopping**: Prevents overfitting
- **Configurable Architecture**: Flexible network dimensions
- **Model Persistence**: Save/load trained networks
- **Multiple Datasets**: XOR, AND, OR logic gate problems

## Building

### Prerequisites
- GCC compiler
- Make (optional, for using Makefile)
- Math library (libm)

### Compilation

Using Makefile:
```bash
make
Manual compilation:
bashgcc -o neural_network src/neural_network.c -lm -Wall -O2
Running
bash./neural_network
Or with make:
bashmake run
Usage
The program automatically runs demonstrations of three logic gate problems:

XOR Problem - Non-linearly separable problem
AND Problem - Simple conjunction
OR Problem - Simple disjunction

Each problem uses different network configurations and activation functions to demonstrate the flexibility of the implementation.