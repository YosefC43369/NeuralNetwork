#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_INPUT_SIZE 10
#define MAX_HIDDEN_SIZE 20
#define MAX_OUTPUT_SIZE 10
#define MAX_EPOCHS 50000
#define CONVERGENCE_THRESHOLD 0.001

// Activation function types
typedef enum {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU
} ActivationType;

/* Neural Network Structure */
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;
    ActivationType activation_type;
    double momentum_factor;

    double WeightsInputHidden[MAX_INPUT_SIZE][MAX_HIDDEN_SIZE];
    double WeightsHiddenOutput[MAX_HIDDEN_SIZE][MAX_OUTPUT_SIZE];
    double BiasHidden[MAX_HIDDEN_SIZE];
    double BiasOutput[MAX_OUTPUT_SIZE];

    double HiddenLayer[MAX_HIDDEN_SIZE];
    double OutputLayer[MAX_OUTPUT_SIZE];

    double MomentumWeightsIh[MAX_INPUT_SIZE][MAX_HIDDEN_SIZE];
    double MomentumWeightsHo[MAX_HIDDEN_SIZE][MAX_OUTPUT_SIZE];
    double MomentumBiasHidden[MAX_HIDDEN_SIZE];
    double MomentumBiasOutput[MAX_OUTPUT_SIZE];

    double TrainingErrorHistory[MAX_EPOCHS];
    int epochsTrained;
    double finalTrainingError;
} NeuralNetwork;

/* Training data structures */
typedef struct {
    double inputS[MAX_INPUT_SIZE];
    double targetS[MAX_OUTPUT_SIZE];
} TrainingExample;

typedef struct {
    TrainingExample* example;
    int numExamples;
    int InputSize;
    int OutputSize;
} Dataset;

/* Function prototypes */
double SigmoID(double x);
double SigmoID_Derivative(double x);
double Activate(double x, ActivationType type);
double ActivateDerivative(double x, ActivationType type);

void initializaNetwork(NeuralNetwork* nn, int input_size, int hidden_size,
                      int output_size, double learning_rate,
                      ActivationType activation_type, double momentum_factor);
void forwardPropagation(NeuralNetwork* nn, double* input);
void backpropagation(NeuralNetwork* nn, double* input, double* target);
void TrainNetwork(NeuralNetwork* nn, Dataset* TrainingDaTa,
                 Dataset* validationData, int MaxEpochs, int verbose);
void TestNetwork(NeuralNetwork* nn, Dataset* TestData);

double CalculateMse(NeuralNetwork* nn, Dataset* dataset);
void SaveNetwork(NeuralNetwork* nn, const char* filename);
void PrintNetworkWeights(NeuralNetwork* nn);

Dataset* createXorDataset(void);
Dataset* createAndDataset(void);
Dataset* createOrDataset(void);
void freeDataset(Dataset* dataset);

#endif  /* NEURAL_NETWORK_H */