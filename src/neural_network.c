#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <errno.h>

#define MAX_INPUT_SIZE 10
#define MAX_HIDDEN_SIZE 20
#define MAX_OUTPUT_SIZE 10
#define MAX_LAYERS 5
#define MAX_EPOCHS 50000
#define DEFAULT_LEARNING_RATE 0.1
#define CONVERGENCE_THRESHOLD 0.001

/* Activation function types */
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

    double WeightsInputHidden[MAX_INPUT_SIZE][MAX_HIDDEN_SIZE];
    double WeightsHiddenOutput[MAX_HIDDEN_SIZE][MAX_OUTPUT_SIZE];
    double BiasHidden[MAX_HIDDEN_SIZE];
    double BiasOutput[MAX_OUTPUT_SIZE];

    double HiddenLayer[MAX_HIDDEN_SIZE];
    double OutputLayer[MAX_OUTPUT_SIZE];

    /* For momentum-based optimization */
    double MomentumWeightsIh[MAX_INPUT_SIZE][MAX_HIDDEN_SIZE];
    double MomentumWeightsHo[MAX_HIDDEN_SIZE][MAX_OUTPUT_SIZE];
    double MomentumBiasHidden[MAX_HIDDEN_SIZE];
    double MomentumBiasOutput[MAX_OUTPUT_SIZE];
    double MomentumFactor;

    /* Training statistics */
    double TrainingErrorHistory[MAX_EPOCHS];
    int epochsTrained;
    double finalTrainingError;
} NeuralNetwork;

/* Training data structure */
typedef struct {
    double inputs[MAX_INPUT_SIZE];
    double targets[MAX_OUTPUT_SIZE];
} TrainingExample;

/* Dataset structure */
typedef struct {
    TrainingExample* examples;
    int numExamples;
    int InputSize;
    int OutputSize;
} Dataset;

/* Activation functions */
double SigmoID(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double SigmoID_Derivative(double x) {
    return x * (1.0 - x);
}

double tanhActivation(double x) {
    return tanh(x);
}

double tanhDerivative(double x) {
    return 1.0 - (x * x);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double reluDerivative(double x) {
    return x > 0 ? x : 0.01 * x;
}

double leakyRelu(double x) {
    return x > 0 ? x : 0.01 * x;
}

double leaky_reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.01;
}

/* Generic activation function */
double activate(double x, ActivationType type) {
    switch (type) {
        case SIGMOID: return SigmoID(x);
        case TANH: return tanhActivation(x);
        case RELU: return relu(x);
        case LEAKY_RELU: return leakyRelu(x);
    }
}

/* Generic activation derivative */
double activateDerivative(double x, ActivationType type) {
    switch (type) {
        case SIGMOID: return SigmoID_Derivative(x);
        case TANH: return tanhDerivative(x);
        case RELU: return reluDerivative(x);
        case LEAKY_RELU: return leaky_reluDerivative(x);
        default: return SigmoID_Derivative(x);
    }
}

/* Initialize network with configurable parameters */
void initializeNetwork(NeuralNetwork* nn, int input_size, int hidden_size, int output_size,
                      double learning_rate, ActivationType activation_type, double MomentumFactor) {
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;
    nn->learning_rate = learning_rate;
    nn->activation_type = activation_type;
    nn->MomentumFactor = MomentumFactor;
    nn->epochsTrained = 0;
    nn->finalTrainingError = 0.0;

    srand(time(NULL));

    /* Xavier initialization for better convergence */
    double xavier_inputHidden = sqrt(6.0 / (input_size + hidden_size));
    double xavier_hiddenOutput = sqrt(6.0 / (hidden_size + output_size));

    /* Initialize weights between input and hidden layer */
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            nn->WeightsInputHidden[i][j] = ((double)rand() / RAND_MAX) * 2.0 * xavier_inputHidden - xavier_inputHidden;
            nn->MomentumWeightsIh[i][j] = 0.0;
        }
    }

    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < output_size; j++) {
            nn->WeightsHiddenOutput[i][j] = ((double)rand() / RAND_MAX) * 2.0 * xavier_hiddenOutput - xavier_hiddenOutput;
            nn->MomentumWeightsHo[i][j] = 0.0;
        }
    }

    for (int i = 0; i < hidden_size; i++) {
        nn->BiasHidden[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        nn->MomentumBiasHidden[i] = 0.0;
    }

    for (int i = 0; i < output_size; i++) {
        nn->BiasOutput[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        nn->MomentumBiasOutput[i] = 0.0;
    }

    for (int i = 0; i < MAX_EPOCHS; i++) {
        nn->TrainingErrorHistory[i] = 0.0;
    }
}

/* Forward propagation */
void forward_propagation(NeuralNetwork* nn, double* input) {
    // Calculate hidden layer
    for (int j = 0; j < nn->hidden_size; j++) {
        double sum = 0.0;
        for (int i = 0; i < nn->input_size; i++) {
            sum += input[i] * nn->WeightsInputHidden[i][j];
        }
        sum += nn->BiasHidden[j];
        nn->HiddenLayer[j] = activate(sum, nn->activation_type);
    }

    /* Calculate output layer */
    for (int j = 0; j < nn->output_size; j++) {
        double sum = 0.0;
        for (int i = 0; i < nn->hidden_size; i++) {
            sum += nn->HiddenLayer[i] * nn->WeightsHiddenOutput[i][j];
        }
        sum += nn->BiasOutput[j];
        nn->OutputLayer[j] = activate(sum, nn->activation_type);
    }
}

/* Backpropagation with momentum */
void backpropation(NeuralNetwork* nn, double* input, double* target) {
    // Calculate output layer error
    double outputErr[MAX_OUTPUT_SIZE];
    double outputDelta[MAX_OUTPUT_SIZE];

    for (int i = 0; i < nn->output_size; i++) {
        outputErr[i] = target[i] - nn->OutputLayer[i];
        outputDelta[i] = outputErr[i] * activateDerivative(nn->OutputLayer[i], nn->activation_type);
    }

    // Calculate hidden layer error
    double hiddenErr[MAX_HIDDEN_SIZE];
    double hiddenDelta[MAX_HIDDEN_SIZE];

    for (int i = 0; i < nn->hidden_size; i++) {
        hiddenErr[i] = 0.0;
        for (int j = 0; j < nn->output_size; j++) {
            hiddenErr[i] += outputDelta[j] * nn->WeightsHiddenOutput[i][j];
        }
        hiddenDelta[i] = hiddenErr[i] * activateDerivative(nn->HiddenLayer[i], nn->activation_type);
    }

    // Update weights and biases (hidden to output) with momentum
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            double WeightChange = nn->learning_rate * outputDelta[j] * nn->HiddenLayer[i];
            nn->MomentumWeightsHo[i][j] = nn->MomentumFactor * nn->MomentumWeightsHo[i][j] + WeightChange;
            nn->WeightsHiddenOutput[i][j] += nn->MomentumWeightsHo[i][j];
        }
    }

    for (int i = 0; i < nn->output_size; i++) {
        double BiasChange = nn->learning_rate * outputDelta[i];
        nn->MomentumBiasOutput[i] = nn->MomentumFactor * nn->MomentumBiasOutput[i] + BiasChange;
        nn->BiasOutput[i] += nn->MomentumBiasOutput[i];
    }

    // Update weights and biases (input to hidden) with momentum
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            double WeightChange = nn->learning_rate * hiddenDelta[j] * input[i];
            nn->MomentumWeightsIh[i][j] = nn->MomentumFactor * nn->MomentumWeightsIh[i][j] + WeightChange;
            nn->WeightsInputHidden[i][j] += nn->MomentumWeightsIh[i][j];
        }
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        double BiasChange = nn->learning_rate * hiddenDelta[i];
        nn->MomentumBiasHidden[i] = nn->MomentumFactor * nn->MomentumBiasHidden[i] + BiasChange;
        nn->BiasHidden[i] += nn->MomentumBiasHidden[i];
    }
}

// Calculate mean squared error
double CalculateMSE(NeuralNetwork* nn, Dataset* dataset) {
    double totalErr = 0.0;

    for (int sample = 0; sample < dataset->numExamples; sample++) {
        forward_propagation(nn, dataset->examples[sample].inputs);

        for (int i = 0; i < nn->output_size; i++) {
            double error = dataset->examples[sample].targets[i] - nn->OutputLayer[i];
            totalErr += error * errno;
        }
    }

    return totalErr / (dataset->numExamples * nn->output_size);
}

void TrainNetwork(NeuralNetwork* nn, Dataset* TrainingData, Dataset* validation_data,
                 int maxEpochs, int verbose) {
    double bestValidationErr = 1e10;
    int patience = 5000;
    int noimprovementCount = 0;

    printf("Training neural network...\n");
    printf("Architecture: %d-%d-%d\n", nn->input_size, nn->hidden_size, nn->output_size);
    printf("Learning rate: %.4f, Momentum: %.4f\n", nn->learning_rate, nn->MomentumFactor);
    printf("Activation function: %s\n",
           nn->activation_type == SIGMOID ? "Sigmoid" :
           nn->activation_type == TANH ? "Tanh" :
           nn->activation_type == RELU ? "ReLU" : "Leaky ReLU");
    printf("----------------------------------------\n");

    for (int epoch = 0; epoch < maxEpochs; epoch++) {
        // Shuffle training data
        for (int i = TrainingData->numExamples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            TrainingExample temp = TrainingData->examples[i];
            TrainingData->examples[i] = TrainingData->examples[j];
            TrainingData->examples[j] = temp;
        }

        /* Train on all examples */
        for (int sample =0; sample < TrainingData->numExamples; sample++) {
            forward_propagation(nn, TrainingData->examples[sample].inputs);
            backpropation(nn, TrainingData->examples[sample].inputs, TrainingData->examples[sample].targets);
        }

        // Calculate training error
        double TrainingErr = CalculateMSE(nn, TrainingData);
        nn->TrainingErrorHistory[epoch] = TrainingErr;

        // Calculate validation error if validation set provided
        double ValidationErr = 0.0;
        if (validation_data != NULL) {
            ValidationErr = CalculateMSE(nn, validation_data);

            // Early stopping check
            if (ValidationErr < bestValidationErr) {
                bestValidationErr = ValidationErr;
                noimprovementCount = 0;
            } else {
                noimprovementCount++;
                if (noimprovementCount >= patience) {
                    printf("Early stopping at epoch %d\n", epoch);
                    break;
                }
            }
        }

        // Print progress
        if (verbose && (epoch % 1000 == 0 || epoch == maxEpochs - 1)) {
            printf("Epoch %d: Training Error = %.6f", epoch, TrainingErr);
            if (validation_data != NULL) {
                printf(", Validation Error = %.6f", ValidationErr);
            }
            printf("\n");
        }

        // Check for convergence
        if (TrainingErr < CONVERGENCE_THRESHOLD) {
            printf("Converged at epoch %d with error %.6f\n", epoch, TrainingErr);
            break;
        }

        nn->epochsTrained = epoch + 1;
    }

    nn->finalTrainingError = nn->TrainingErrorHistory[nn->epochsTrained - 1];
    printf("Training completed after %d epochs\n", nn->epochsTrained);
    printf("Final training error: %.6f\n", nn->finalTrainingError);
}

/* Test the network */
void TestNetwork(NeuralNetwork* nn, Dataset* TestData) {
    printf("\nTesting the network:\n");
    printf("Sample\t");
    for (int i = 0; i < nn->input_size; i++) {
        printf("Input%d\t", i + 1);
    }
    for (int i = 0; i < nn->output_size; i++) {
        printf("Output%d\t", i + 1);
    }
    for (int i = 0; i < nn->output_size; i++) {
        printf("Expected%d\t", i + 1);
    }
    printf("Error\n");

    printf("------\t");
    for (int i = 0; i < nn->input_size + 2 * nn->output_size + 1; i++) {
        printf("-------\t");
    }
    printf("\n");

    double totalErr = 0.0;
    for (int sample = 0; sample < TestData->numExamples; sample++) {
        forward_propagation(nn, TestData->examples[sample].inputs);

        printf("%d\t", sample + 1);
        for (int i = 0; i < nn->input_size; i++) {
            printf("%.1f\t", TestData->examples[sample].inputs[i]);
        }
        for (int i = 0; i < nn->output_size; i++) {
            printf("%.4f\t", nn->OutputLayer[i]);
        }
        for (int i = 0; i < nn->output_size; i++) {
            printf("%.1f\t", TestData->examples[sample].targets[i]);
        }

        double sampleErr = 0.0;
        for (int i = 0; i < nn->output_size; i++) {
            double error = fabs(TestData->examples[sample].targets[i] - nn->OutputLayer[i]);
            sampleErr += error;
        }
        totalErr += sampleErr;
        printf("%.4f\n", sampleErr);
    }

    printf("\nAverage error: %.4f\n", totalErr / TestData->numExamples);
    printf("Test accuracy: %.2f%%\n", (1.0 - totalErr / TestData->numExamples) * 100.0);
}

/* Create XOR dataset */
Dataset* createXORDataset() {
    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->numExamples = 4;
    dataset->InputSize = 2;
    dataset->OutputSize = 1;
    dataset->examples = malloc(4 * sizeof(TrainingExample));

    // XOR truth table
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4][1] = {{0}, {1}, {1}, {0}};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            dataset->examples[i].inputs[j] = inputs[i][j];
        }
        dataset->examples[i].targets[0] = targets[i][0];
    }

    return dataset;
}

/* Create AND dataset */
Dataset* CreateANDDataset() {

    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->numExamples = 4;
    dataset->InputSize = 2;
    dataset->OutputSize = 1;
    dataset->examples = malloc(4 * sizeof(TrainingExample));

    // AND truth table
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4][1] = {{0}, {0}, {0}, {1}};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            dataset->examples[i].inputs[j] = inputs[i][j];
        }
        dataset->examples[i].targets[0] = targets[i][0];
    }

    return dataset;
}

/* Create OR dataset */
Dataset* CreateORDataset() {
    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->numExamples = 4;
    dataset->InputSize = 2;
    dataset->InputSize = 1;
    dataset->examples = malloc(4 * sizeof(TrainingExample));

    // OR truth table
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4][1] = {{0}, {1}, {1}, {1}};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            dataset->examples[i].inputs[j] = inputs[i][j];
        }
        dataset->examples[i].targets[0] = targets[i][0];
    }

    return dataset;
}

/* Print network weights for debugging */
void PrintNetworkWeights(NeuralNetwork* nn) {
    printf("\nNetwork Weights:\n");
    printf("Input to Hidden:\n");
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            printf("%.4f ", nn->WeightsInputHidden[i][j]);
        }
        printf("\n");
    }

    printf("Hidden to Output:\n");
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            printf("%.4f ", nn->WeightsHiddenOutput[i][j]);
        }
        printf("\n");
    }
}

/* Save network to file */
void SaveNetwork(NeuralNetwork* nn, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Cannot open file %s for writing\n", filename);
        return;
    }

    fprintf(file, "%d %d %d\n", nn->input_size, nn->hidden_size, nn->output_size);
    fprintf(file, "%.6f %.6f %d\n", nn->learning_rate, nn->MomentumFactor, nn->activation_type);

    // Save weights
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            fprintf(file, "%.6f", nn->WeightsInputHidden[i][j]);
        }
        fprintf(file, "\n");
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            fprintf(file, "%.6f ", nn->WeightsHiddenOutput[i][j]);
        }
        fprintf(file, "\n");
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        fprintf(file, "%.6f ", nn->BiasHidden[i]);
    }
    fprintf(file,"\n");

    fclose(file);
    printf("Network saved to %s\n", filename);
}

/* Free dataset memory */
void free_dataset(Dataset* dataset) {
    free(dataset->examples);
    free(dataset);
}

void RunDemo() {
    printf("=== Neural Network Demo ===\n\n");

    // Test XOR problem
    printf("1. XOR Problem:\n");
    Dataset* XORData = createXORDataset();
    NeuralNetwork xor_nn;
    initializeNetwork(&xor_nn, 2, 4, 1, 0.3, SIGMOID, 0.9);
    TrainNetwork(&xor_nn, XORData, NULL, 10000, 1);
    TestNetwork(&xor_nn, XORData);
    SaveNetwork(&xor_nn, "xor_network.txt");
    free_dataset(XORData);

    printf("\n");
    for (int i = 0; i < 50; i++) printf("=");
    printf("\n");

    // Test AND problem
    printf("2. AND Problem:\n");
    Dataset* and_data = CreateANDDataset();
    NeuralNetwork and_nn;
    initializeNetwork(&and_nn, 2, 3, 1, 0.5, TANH, 0.8);
    TrainNetwork(&and_nn, and_data, NULL, 5000, 1);
    TestNetwork(&and_nn, and_data);
    free_dataset(and_data);

    printf("\n");
    for (int i = 0; i < 50; i++) printf("=");
    printf("\n");

    // Test OR problem
    printf("3. OR Problem:\n");
    Dataset* or_data = createXORDataset();
    NeuralNetwork or_nn;
    initializeNetwork(&or_nn, 2, 3, 1, 0.4, RELU, 0.7);
    TrainNetwork(&or_nn, or_data, NULL, 5000, 1);
    TestNetwork(&or_nn, or_data);
    free_dataset(or_data);
}

int main() {
    printf("Advanced Neural Network with Backpropagation in C\n");
    printf("================================================\n\n");

    RunDemo();

    printf("\nDemo completed!\n");
    printf("The neural networks have learned to solve logic gate problems.\n");
    printf("Features demonstrated:\n");
    printf("- Multiple activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU)\n");
    printf("- Momentum-based optimization\n");
    printf("- Xavier weight initialization\n");
    printf("- Early stopping and convergence detection\n");
    printf("- Network serialization\n");
    printf("- Configurable architecture\n");

    return 0;
}