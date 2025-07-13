#include "activation.h"
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

/*
 * =============================================================================
 * ACTIVATION FUNCTIONS IMPLEMENTATION
 * =============================================================================
 * 
 * This file implements comprehensive activation functions for neural networks.
 * It provides vectorized operations, parameter validation, and thread-safe
 * implementations for various activation functions.
 * 
 * Author: YosefKlinhom
 * Version: 2.0
 * Date: 2025
 * 
 * =============================================================================
 */

/* =============================================================================
 * GLOBAL VARIABLES AND CONSTANTS
 * ============================================================================= */

/* SELU constants */
static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

/* Error handler callback */
static void (*error_handler)(int error_code, const char* message) = NULL;

/* Activation function information table */
static const ActivationInfo activation_info_table[] = {
    {ACTIVATION_SIGMOID, "Sigmoid", "1/(1+e^(-x))", "Sigmoid activation function", 0.0, 1.0, 1, 1, 1, 1, 0, "Binary classification, output layers"},
    {ACTIVATION_TANH, "Tanh", "tanh(x)", "Hyperbolic tangent activation", -1.0, 1.0, 1, 1, 1, 1, 0, "Hidden layers, RNNs"},
    {ACTIVATION_RELU, "ReLU", "max(0,x)", "Rectified Linear Unit", 0.0, INFINITY, 0, 1, 0, 0, 0, "Hidden layers, CNNs"},
    {ACTIVATION_LEAKY_RELU, "Leaky ReLU", "max(αx,x)", "Leaky Rectified Linear Unit", -INFINITY, INFINITY, 0, 1, 1, 0, 1, "Hidden layers, avoid dying neurons"},
    {ACTIVATION_ELU, "ELU", "x if x>0, α(e^x-1) if x≤0", "Exponential Linear Unit", -INFINITY, INFINITY, 0, 1, 1, 0, 1, "Hidden layers, smooth negative part"},
    {ACTIVATION_SELU, "SELU", "λ*ELU(x,α)", "Scaled Exponential Linear Unit", -INFINITY, INFINITY, 0, 1, 1, 0, 0, "Self-normalizing networks"},
    {ACTIVATION_SWISH, "Swish", "x*sigmoid(βx)", "Swish activation function", -INFINITY, INFINITY, 0, 0, 1, 0, 1, "Modern architectures, smooth"},
    {ACTIVATION_MISH, "Mish", "x*tanh(softplus(x))", "Mish activation function", -INFINITY, INFINITY, 0, 0, 1, 0, 0, "Modern architectures, smooth"},
    {ACTIVATION_GELU, "GELU", "0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))", "Gaussian Error Linear Unit", -INFINITY, INFINITY, 0, 0, 1, 0, 0, "Transformers, modern NLP"},
    {ACTIVATION_STEP, "Step", "1 if x>0, 0 if x≤0", "Step function", 0.0, 1.0, 1, 1, 0, 0, 1, "Binary classification, thresholding"},
    {ACTIVATION_LINEAR, "Linear", "x", "Linear activation", -INFINITY, INFINITY, 0, 1, 1, 0, 0, "Regression, output layers"},
    {ACTIVATION_SOFTPLUS, "Softplus", "ln(1+e^x)", "Softplus activation", 0.0, INFINITY, 0, 1, 1, 0, 0, "Smooth approximation to ReLU"},
    {ACTIVATION_SOFTSIGN, "Softsign", "x/(1+|x|)", "Softsign activation", -1.0, 1.0, 1, 1, 1, 0, 0, "Alternative to tanh"},
    {ACTIVATION_EXPONENTIAL, "Exponential", "e^x", "Exponential function", 0.0, INFINITY, 0, 1, 1, 0, 0, "Specialized applications"},
    {ACTIVATION_HARD_SIGMOID, "Hard Sigmoid", "max(0,min(1,0.2*x+0.5))", "Hard sigmoid approximation", 0.0, 1.0, 1, 1, 0, 0, 0, "Fast sigmoid approximation"},
    {ACTIVATION_HARD_TANH, "Hard Tanh", "max(-1,min(1,x))", "Hard tanh approximation", -1.0, 1.0, 1, 1, 0, 0, 0, "Fast tanh approximation"},
    {ACTIVATION_SOFTMAX, "Softmax", "e^xi/Σe^xj", "Softmax function", 0.0, 1.0, 1, 0, 1, 0, 0, "Multi-class classification"},
    {ACTIVATION_LOG_SOFTMAX, "Log Softmax", "ln(e^xi/Σe^xj)", "Log softmax function", -INFINITY, 0.0, 0, 0, 1, 0, 0, "Multi-class with log probabilities"},
    {ACTIVATION_CUSTOM, "Custom", "user-defined", "Custom activation function", -INFINITY, INFINITY, 0, 0, 0, 0, 1, "Specialized applications"}
};

/* =============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================= */

double activation_clip(double x, double min_val, double max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

double activation_safe_exp(double x) {
    if (x > ACTIVATION_CLIP_VALUE) return exp(ACTIVATION_CLIP_VALUE);
    if (x < -ACTIVATION_CLIP_VALUE) return exp(-ACTIVATION_CLIP_VALUE);
    return exp(x);
}

double activation_safe_log(double x) {
    if (x <= 0.0) return log(ACTIVATION_EPSILON);
    return log(x);
}

int activation_is_valid(double x) {
    return !isnan(x) && !isinf(x);
}

/* =============================================================================
 * ERROR HANDLING
 * ============================================================================= */

 const char* activation_get_error_message(int error_code) {
    switch (error_code) {
        case ACTIVATION_SUCCESS: return "Success";
        case ACTIVATION_ERROR_NULL: return "Null pointer error";
        case ACTIVATION_ERROR_SIZE: return "Invalid size error";
        case ACTIVATION_ERROR_PARAM: return "Invalid parameter error";
        case ACTIVATION_ERROR_RANGE: return "Value out of range error";
        default: return "Unknown error";
    }
 }

 void activation_set_error_handler(void (*handler)(int error_code, const char* message)) {
    error_handler = handler;
}

static void report_error(int error_code) {
    if (error_handler) {
        error_handler(error_code, activation_get_error_message(error_code));
    }
}

/* =============================================================================
 * BASIC ACTIVATION FUNCTIONS
 * ============================================================================= */

double activation_sigmoid(double x, const ActivationParams* params) {
    x = activation_clip(x, -ACTIVATION_CLIP_VALUE, ACTIVATION_CLIP_VALUE);
    return 1.0 / (1.0 + exp(-x));
}

double activation_sigmoid_derivative(double x, const ActivationParams* params) {
    return x * (1.0 - x);
}

double activation_tanh(double x, const ActivationParams* params) {
    x = activation_clip(x, -ACTIVATION_CLIP_VALUE, ACTIVATION_CLIP_VALUE);
    return tahn(x);
}

double activation_tanh_derivative(double x, const ActivationParams* params) {
    return 1.0 - (x * x);
}

double activation_relu(double x, const ActivationParams* params) {
    return x > 0.0 ? x : 0.0;
}

double activation_relu_derivative(double x, const ActivationParams* params) {
    return x > 0.0 ? 1.0 : 0.0;
}

double activation_leaky_relu(double x, const ActivationParams* params) {
    double alpha = (params && params->alpha != 0.0) ? params->alpha : 0.01;
    return x > 0.0 ? x : alpha * x;
}

double activation_leaky_relu_derivative(double x, const ActivationParams* params) {
    double alpha = (params && params->alpha != 0.0) ? params->alpha : 0.01;
    return x > 0.0 ? 1.0 : alpha;
}

/* =============================================================================
 * ADVANCED ACTIVATION FUNCTIONS
 * ============================================================================= */

double activation_elu(double x, const ActivationParams* params) {
    double alpha = (params && params->alpha != 0.0) ? params->alpha : 1.0;
    if (x > 0.0) return x;
    return alpha * (activation_safe_exp(x) - 1.0);
}

double activation_elu_derivative(double x, const ActivationParams* params) {
    double alpha = (params && params->alpha != 0.0) ? params->alpha : 1.0;
    if (x > 0.0) return 1.0;
    return alpha * activation_safe_exp(x);
}

double activation_selu(double x, const ActivationParams* params) {
    if (x > 0.0) return SELU_SCALE * x;
    return SELU_SCALE * SELU_ALPHA * (activation_safe_exp(x) - 1.0);
}

double activation_selu_derivative(double x, const ActivationParams* params) {
    if (x > 0.0) return SELU_SCALE;
    return SELU_SCALE * SELU_ALPHA * activation_safe_exp(x);
}

double activation_swish(double x, const ActivationParams* params) {
    double beta = (params && params->beta != 0.0) ? params->beta : 1.0;
    return x * activation_sigmoid(beta * x, NULL);
}

double activation_swish_derivative(double x, const ActivationParams* params) {
    double beta = (params && params->beta != 0.0) ? params->beta : 1.0;
    double sigmoid_val = activation_sigmoid(beta * x, NULL);
    return sigmoid_val + x * beta * sigmoid_val * (1.0 - sigmoid_val);
}

double activation_mish(double x, const ActivationParams* params) {
    double softplus_val = activation_softplus(x, NULL);
    return x * tanh(softplus_val);
}

double activation_mish_derivative(double x, const ActivationParams* params) {
    double sigmoid_val = activation_sigmoid(x, NULL);
    double tanh_softplus = tanh(activation_softplus(x, NULL));
    double sech_softplus = 1.0 - tanh_softplus * tanh_softplus;
    return tanh_softplus + x * sigmoid_val * sech_softplus;
}

double activation_gelu(double x, const ActivationParams* params) {
    double sqrt_2_pi = sqrt(2.0 / ACTIVATION_PI);
    double tanh_arg = sqrt_2_pi * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(tanh_arg));
}

double activation_gelu_derivative(double x, const ActivationParams* params) {
    double sqrt_2_pi = sqrt(2.0 / ACTIVATION_PI);
    double tanh_arg = sqrt_2_pi * (x + 0.044715 * x * x * x);
    double tanh_val = tanh(tanh_arg);
    double sech_sq = 1.0 - tanh_val * tanh_val;
    double dtanh_dx = sqrt_2_pi * (1.0 + 0.134145 * x * x);
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech_sq * dtanh_dx;
}

/* =============================================================================
 * UTILITY ACTIVATION FUNCTIONS
 * ============================================================================= */

 double activation_step(double x, const ActivationParams* params) {
    double threshold = (params) ? params->threshold : 0.0;
    return x > threshold ? 1.0 : 0.0;
 }

 double activation_linear(double x, const ActivationParams* params) {
    double scale = (params && params->scale != 0.0) ? params->scale : 1.0;
    double offset = (params) ? params->offset : 0.0;
    return scale * x + offset;
}

double activation_linear_derivative(double x, const ActivationParams* params) {
    double scale = (params && params->scale != 0.0) ? params->scale : 1.0;
    return scale;
}

double activation_softplus(double x, const ActivationParams* params) {
    if (x > ACTIVATION_CLIP_VALUE) return x;
    return log(1.0 + activation_safe_exp(x));
}

double activation_softplus_derivative(double x, const ActivationParams* params) {
    return activation_sigmoid(x, NULL);
}

double activation_softsign(double x, const ActivationParams* params) {
    return x / (1.0 + fabs(x));
}

double activation_softsign_derivative(double x, const ActivationParams* params) {
    double abs_x = fabs(x);
    double denom = 1.0 + abs_x;
    return 1.0 / (denom * denom);
}

/* =============================================================================
 * HARD APPROXIMATION FUNCTIONS
 * ============================================================================= */

 double activation_hard_sigmoid(double x, const ActivationParams* params) {
    return activation_clip(0.2 * x + 0.5, 0.0, 1.0);
}

double activation_hard_sigmoid_derivative(double x, const ActivationParams* params) {
    return (x > -2.5 && x < 2.5) ? 0.2 : 0.0;
}

double activation_hard_tanh(double x, const ActivationParams* params) {
    return activation_clip(x, -1.0, 1.0);
}

double activation_hard_tanh_derivative(double x, const ActivationParams* params) {
    return (x > -1.0 && x < 1.0) ? 1.0 : 0.0;
}

/* =============================================================================
 * SOFTMAX AND LOG-SOFTMAX FUNCTIONS
 * ============================================================================= */

int activation_softmax(const double* input, double* output, int size, const ActivationParams* params) {
    if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    if (size > MAX_VECTOR_SIZE) {
        report_error(ACTIVATION_ERROR_SIZE);
        return ACTIVATION_ERROR_SIZE;
    }

    // Find maximum for numerical stability
    double max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exponentials and sum
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = activation_safe_exp(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    if (sum > ACTIVATION_EPSILON) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    }

    return ACTIVATION_SUCCESS;
}

int activation_log_softmax(const double* input, double* output, int size, const ActivationParams* params) {
    if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    if (size > MAX_VECTOR_SIZE) {
        report_error(ACTIVATION_ERROR_SIZE);
        return ACTIVATION_ERROR_SIZE;
    }

    // Find maximum for numerical stability
    double max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute log-sum-exp
    double log_sum = 0.0;
    for (int i = 0; i < size; i++) {
        log_sum += activation_safe_exp(input[i] - max_val);
    }
    log_sum = max_val + log(log_sum);

    // Compute log_softmax
    for (int i = 0; i < size; i++) {
        output[i] = input[i] - log_sum;
    }

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * CUSTOM ACTIVATION FUNCTIONS
 * ============================================================================= */

double activation_custom(double x, const ActivationParams *params) {
    if (!params || !params->custom_func) {
        report_error(ACTIVATION_ERROR_PARAM);
        return 0.0;
    }
    return params->custom_func(x, params->custom_params);
}

double activation_custom_derivative(double x, const ActivationParams* params) {
    if (!params || !params->custom_deriv) {
        report_error(ACTIVATION_ERROR_PARAM);
        return 0.0;
    }
    return params->custom_deriv(x, params->custom_params);
}

/* =============================================================================
 * GENERIC ACTIVATION INTERFACE
 * ============================================================================= */

double activation_apply(ActivationType type, double x, const ActivationParams* params) {
    switch (type) {
        case ACTIVATION_SIGMOID: return activation_sigmoid(x, params);
        case ACTIVATION_TANH: return activation_tanh(x, params);
        case ACTIVATION_RELU: return activation_relu(x, params);
        case ACTIVATION_LEAKY_RELU: return activation_leaky_relu(x, params);
        case ACTIVATION_ELU: return activation_elu(x, params);
        case ACTIVATION_SELU: return activation_selu(x, params);
        case ACTIVATION_SWISH: return activation_swish(x, params);
        case ACTIVATION_MISH: return activation_mish(x, params);
        case ACTIVATION_GELU: return activation_gelu(x, params);
        case ACTIVATION_STEP: return activation_step(x, params);
        case ACTIVATION_LINEAR: return activation_linear(x, params);
        case ACTIVATION_SOFTPLUS: return activation_softplus(x, params);
        case ACTIVATION_SOFTSIGN: return activation_softsign(x, params);
        case ACTIVATION_EXPONENTIAL: return activation_safe_exp(x);
        case ACTIVATION_HARD_SIGMOID: return activation_hard_sigmoid(x, params);
        case ACTIVATION_HARD_TANH: return activation_hard_tanh(x, params);
        case ACTIVATION_CUSTOM: return activation_custom(x, params);
        default:
            report_error(ACTIVATION_ERROR_PARAM);
            return 0.0;
    }
}

double activation_apply_derivative(ActivationType type, double x, const ActivationParams* params) {
    switch (type) {
        case ACTIVATION_SIGMOID: return activation_sigmoid_derivative(x, params);
        case ACTIVATION_TANH: return activation_tanh_derivative(x, params);
        case ACTIVATION_RELU: return activation_relu_derivative(x, params);
        case ACTIVATION_LEAKY_RELU: return activation_leaky_relu_derivative(x, params);
        case ACTIVATION_ELU: return activation_elu_derivative(x, params);
        case ACTIVATION_SELU: return activation_selu_derivative(x, params);
        case ACTIVATION_SWISH: return activation_swish_derivative(x, params);
        case ACTIVATION_MISH: return activation_mish_derivative(x, params);
        case ACTIVATION_GELU: return activation_gelu_derivative(x, params);
        case ACTIVATION_LINEAR: return activation_linear_derivative(x, params);
        case ACTIVATION_SOFTPLUS: return activation_softplus_derivative(x, params);
        case ACTIVATION_SOFTSIGN: return activation_softsign_derivative(x, params);
        case ACTIVATION_EXPONENTIAL: return activation_safe_exp(x);
        case ACTIVATION_HARD_SIGMOID: return activation_hard_sigmoid_derivative(x, params);
        case ACTIVATION_HARD_TANH: return activation_hard_tanh_derivative(x, params);
        case ACTIVATION_CUSTOM: return activation_custom_derivative(x, params);
        default:
            report_error(ACTIVATION_ERROR_PARAM);
            return 0.0;
    }
}

/* =============================================================================
 * VECTORIZED OPERATIONS
 * ============================================================================= */

double activation_apply_vector(ActivationType type, const double* input, double* output,
                              int size, const ActivationParams* params) {
       if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
       }

       if (size > MAX_VECTOR_SIZE) {
        report_error(ACTIVATION_ERROR_SIZE);
        return ACTIVATION_ERROR_SIZE;
       }

       // Handle softmax and log-softmax separately
       if (type == ACTIVATION_SOFTMAX) {
        return activation_softmax(input, output, size, params);
       }
       if (type == ACTIVATION_LOG_SOFTMAX) {
        return activation_log_softmax(input, output, size, params);
       }

       // Apply activation function element-wise
       for (int i = 0; i < size; i++) {
        output[i] = activation_apply(type, input[i], params);
       }

       return ACTIVATION_SUCCESS;
}

int activation_apply_derivative_vector(ActivationType type, const double* input, double* output, 
                                      int size, const ActivationParams* params) {
    if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }
    
    if (size > MAX_VECTOR_SIZE) {
        report_error(ACTIVATION_ERROR_SIZE);
        return ACTIVATION_ERROR_SIZE;
    }

    // Apply derivative function element-wise
    for (int i = 0; i < size; i++) {
        output[i] = activation_apply_derivative(type, input[i], params);
    }

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * PARAMETER MANAGEMENT
 * ============================================================================= */

int activation_init_params(ActivationParams* params, ActivationType type) {
    if (!params) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    // Initialize all parameters to default values
    params->alpha = 0.0;
    params->beta = 0.0;
    params->gamma = 0.0;
    params->lambda = 0.0;
    params->threshold = 0.0;
    params->scale = 1.0;
    params->offset = 0.0;
    params->custom_func = NULL;
    params->custom_deriv = NULL;
    params->custom_params = NULL;

    // Set type-specific defaults
    switch (type) {
        case ACTIVATION_LEAKY_RELU:
            params->alpha = 0.01;
            break;
        case ACTIVATION_ELU:
            params->alpha = 1.0;
            break;
        case ACTIVATION_SWISH:
            params->beta = 1.0;
            break;
        default:
            break;
    }

    return ACTIVATION_SUCCESS;
}

int activation_set_param(ActivationParams* params, const char* param_name, double value) {
    if (!params || !param_name) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    if (strcmp(param_name, "alpha") == 0) {
        params->alpha = value;
    } else if (strcmp(param_name, "beta") == 0) {
        params->beta = value;
    } else if (strcmp(param_name, "gamma") == 0) {
        params->gamma = value;
    } else if (strcmp(param_name, "lambda") == 0) {
        params->lambda = value;
    } else if (strcmp(param_name, "threshold") == 0) {
        params->threshold = value;
    } else if (strcmp(param_name, "scale") == 0) {
        params->scale = value;
    } else if (strcmp(param_name, "offset") == 0) {
        params->offset = value;
    } else {
        report_error(ACTIVATION_ERROR_PARAM);
        return ACTIVATION_ERROR_PARAM;
    }

    return ACTIVATION_SUCCESS;
}

int activation_get_param(const ActivationParams* params, const char* param_name, double* value) {
    if (!params || !param_name || !value) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    if (strcmp(param_name, "alpha") == 0) {
        *value = params->alpha;
    } else if (strcmp(param_name, "beta") == 0) {
        *value = params->beta;
    } else if (strcmp(param_name, "gamma") == 0) {
        *value = params->gamma;
    } else if (strcmp(param_name, "lambda") == 0) {
        *value = params->lambda;
    } else if (strcmp(param_name, "threshold") == 0) {
        *value = params->threshold;
    } else if (strcmp(param_name, "scale") == 0) {
        *value = params->scale;
    } else if (strcmp(param_name, "offset") == 0) {
        *value = params->offset;
    } else {
        report_error(ACTIVATION_ERROR_PARAM);
        return ACTIVATION_ERROR_PARAM;
    }

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * INFORMATION AND UTILITIES
 * ============================================================================= */

const ActivationInfo* activation_get_info(ActivationType type) {
    if (type >= 0 && type < ACTIVATION_COUNT) {
        return &activation_info_table[type];
    }
    return NULL;
}

const char* activation_get_name(ActivationType type) {
    const ActivationInfo* info = activation_get_info(type);
    return info ? info->name : NULL;
}

ActivationType activation_get_type_from_name(const char* name) {
    if (!name) return ACTIVATION_COUNT;

    for (int i = 0; i < ACTIVATION_COUNT; i++) {
        if (strcmp(activation_info_table[i].name, name) == 0) {
            return (ActivationType)i;
        }
    }
    return ACTIVATION_COUNT;
}

int activation_is_bounded(ActivationType type) {
    const ActivationInfo* info = activation_get_info(type);
    return info ? info->is_bounded : -1;
}

int activation_get_range(ActivationType type, double* min_val, double* max_val) {
    const ActivationInfo* info = activation_get_info(type);
    if (!info || !min_val || !max_val) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    *min_val = info->range_min;
    *max_val = info->range_max;
    return ACTIVATION_SUCCESS;
}

void activation_print_info(ActivationType type, FILE* file) {
    if (!file) file = stdout;

    const ActivationInfo* info = activation_get_info(type);
    if (!info) {
        fprintf(file, "Invalid activation type: %d\n", type);
        return;
    }

    fprintf(file, "Activation Function: %s\n", info->name);
    fprintf(file, "Formula: %s\n", info->formula);
    fprintf(file, "Description: %s\n", info->description);
    fprintf(file, "Range: [%.2f, %.2f]\n", info->range_min, info->range_max);
    fprintf(file, "Properties: %s%s%s%s\n",
            info->is_bounded ? "Bounded " : "",
            info->is_monotonic ? "Monotonic " : "",
            info->is_differentiable ? "Differentiable ": "",
            info->has_vanishing_gradient ? "Vanishing-Gradient " : "");
    fprintf(file, "Recommended use: %s\n", info->recommended_use);
    fprintf(file, "\n");
}

void activation_printf_all(FILE* file) {
    if (!file) file = stdout;

    fprintf(file, "Available Activation Functions:\n");
    fprintf(file, "===============================\n\n");

    for (int i = 0; i < ACTIVATION_COUNT; i++) {
        activation_print_info((ActivationType)i, file);
    }
}

/* =============================================================================
 * PARAMETER VALIDATION
 * ============================================================================= */

int activation_validate_params(ActivationType type, const ActivationParams* params) {
    if (!params && (type == ACTIVATION_LEAKY_RELU || type == ACTIVATION_ELU ||
                    type == ACTIVATION_SWISH || type == ACTIVATION_CUSTOM)) {
        report_error(ACTIVATION_ERROR_PARAM);
        return ACTIVATION_ERROR_PARAM;
    }

    if (params) {
        switch (type) {
            case ACTIVATION_LEAKY_RELU:
                if (params->alpha <= 0.0 || params->alpha >= 1.0) {
                    report_error(ACTIVATION_ERROR_RANGE);
                    return ACTIVATION_ERROR_RANGE;
                }
                break;
            case ACTIVATION_ELU:
                if (params->alpha <= 0.0) {
                    report_error(ACTIVATION_ERROR_RANGE);
                    return ACTIVATION_ERROR_RANGE;
                }
                break;
            case ACTIVATION_SWISH:
                if (params->beta <= 0.0) {
                    report_error(ACTIVATION_ERROR_RANGE);
                    return ACTIVATION_ERROR_RANGE;
                }
                break;
            case ACTIVATION_CUSTOM:
                if (params->custom_func) {
                    report_error(ACTIVATION_ERROR_PARAM);
                    return ACTIVATION_ERROR_PARAM;
                }
                break;
            default:
                break;
        }
    }

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * BENCHMARKING AND TESTING
 * ============================================================================= */

double activation_benchmark(ActivationType type, int iterations,
                            const ActivationParams* params) {
    if (iterations <= 0) return -1.0;

    clock_t start = clock();
    double x = 0.5;

    for (int i = 0; i < iterations; i++) {
        activation_apply(type, x, params);
        x = (x + 0.1 > 5.0) ? -5.0 : x + 0.1;
    }

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int activation_test(ActivationType type, const ActivationParams* params, double tolerance) {
    if (tolerance <= 0.0) tolerance = 1e-6;

    double test_values[] = {-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < num_tests; i++) {
        double x = test_values[i];
        double result = activation_apply(type, x, params);

        if (!activation_is_valid(result)) {
            return ACTIVATION_ERROR_RANGE;
        }

        // Test derivative if available
        if (type != ACTIVATION_STEP && type != ACTIVATION_SOFTMAX &&
            type != ACTIVATION_LOG_SOFTMAX) {
            double deriv = activation_apply_derivative(type, x, params);
            if (!activation_is_valid(deriv)) {
                return ACTIVATION_ERROR_RANGE;
            }
        }
    }

    return ACTIVATION_SUCCESS;
}

int activation_test_all(double tolerance) {
    int failed_tests = 0;

    for (int type = 0; type < ACTIVATION_COUNT - 1; type++) {
        ActivationParams params;
        activation_init_params(&params, (ActivationType)type);

        if (activation_test((ActivationType)type, &params, tolerance) != ACTIVATION_SUCCESS) {
            failed_tests++;
        }
    }

    return failed_tests;
}

/* =============================================================================
 * CUSTOM ACTIVATION FUNCTIONS
 * ============================================================================= */

int activation_register_custom(ActivationFunc func, ActivationDerivFunc deriv_func,
                              const char* name, void* params) {
    if (!func || !name) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    // In a full implementation, you would maintain a registry of custom functions
    // For this example, we'll just validate the inputs
    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * THREAD SAFETY
 * ============================================================================= */

static int thread_safe_initialized = 0;

int activation_init_thread_safe(void) {
    if (thread_safe_initialized) return ACTIVATION_SUCCESS;

    // Initialize any thread-local storage or mutexes here
    thread_safe_initialized = 1;
    return ACTIVATION_SUCCESS;
}

void activation_cleanup_thread_safe(void) {
    if (!thread_safe_initialized) return;

    // Cleanup thread-local storage or mutexes here
    thread_safe_initialized = 0;
}

/* =============================================================================
 * ADDITIONAL UTILITY FUNCTIONS
 * ============================================================================= */

/**
 * @brief Exponential activation function
 */
double activation_exponential(double x, const ActivationParams* params) {
    return activation_safe_exp(x);
}

/**
 * @brief Batch normalization helper
 */
int activation_batch_normalize(double* input, double* output, int size,
                              double mean, double variance, double epsilon) {
    if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    if (variance < 0.0) {
        report_error(ACTIVATION_ERROR_RANGE);
        return ACTIVATION_ERROR_RANGE;
    }

    double std_dev = sqrt(variance + epsilon);

    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) / std_dev;
    }

    return ACTIVATION_SUCCESS;
}

/**
 * @brief Layer normalization helper
 */
int activation_layer_normalize(double* input, double* output, int size, double epsilon) {
    if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    // Compute mena
    double mean = 0.0;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;

    // Compute variance
    double variance = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;

    // Normalize
    double std_dev = sqrt(variance + epsilon);
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) / std_dev;
    }

    return ACTIVATION_SUCCESS;
}

/**
 * @brief Gradient clipping utility
 */
int activation_clip_gradients(double* gradients, int size, double max_norm) {
    if (!gradients || size <= 0 || max_norm <= 0.0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    // Compute gradient norm
    double norm = 0.0;
    for (int i = 0; i < size; i++) {
        norm += gradients[i] * gradients[i];
    }
    norm = sqrt(norm);

    // Clip if necessary
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < size; i++) {
            gradients[i] *= scale;
        }
    }

    return ACTIVATION_SUCCESS;
}

/**
 * @brief Activation function comparison utility
 */
int activation_compare_functions(ActivationType type1, ActivationType type2,
                                const ActivationParams* params1,
                                const ActivationParams* params2,
                                double* input, int size, double* mse) {
    if (!input || !mse || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    double* output1 = malloc(size * sizeof(double));
    double* output2 = malloc(size * sizeof(double));

    if (!output1 || !output2) {
        free(output1);
        free(output2);
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    // Apply both activation function
    activation_apply_vector(type1, input, output1, size, params1);
    activation_apply_vector(type2, input, output2, size, params2);

    // Compute MSE
    double sum_sq_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = output1[i] - output2[i];
        sum_sq_diff += diff * diff;
    }
    *mse = sum_sq_diff / size;

    free(output1);
    free(output2);

    return ACTIVATION_SUCCESS;
}

/**
 * @brief Activation function statistics
 */
int activation_compute_stats(const double* input, int size, double* mean,
                            double* variance, double* min_val, double* max_val) {
    if (!input || size <= 0 || !mean || !variance || !min_val || !max_val) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    // Initialize
    *mean = 0.0;
    *variance = 0.0;
    *min_val = input[0];
    *max_val = input[0];

    // Compute mean and find min/max
    for (int i = 0; i < size; i++) {
        *mean += input[i];
        if (input[i] < *min_val) *min_val = input[i];
        if (input[i] > *max_val) *max_val = input[i];
    }
    *mean /= size;

    // Compute variance
    for (int i = 0; i < size; i++) {
        double diff = input[i] - *mean;
        *variance += diff * diff;
    }
    *variance /= size;

    return ACTIVATION_SUCCESS;
}

/**
 * @brief Save activation function configuration
 */
int activation_save_config(const char* filename, ActivationType type,
                          const ActivationParams* params) {
    if (!filename) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    FILE* file = fopen(filename, "w");
    if (!file) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    fprintf(file, "activation_type=%d\n", type);
    if (!params) {
        fprintf(file, "alpha=%f\n", params->alpha);
        fprintf(file, "beta=%f\n", params->beta);
        fprintf(file, "gamma=%f\n", params->gamma);
        fprintf(file, "lambda=%f\n", params->lambda);
        fprintf(file, "threshold=%f\n", params->threshold);
        fprintf(file, "scale=%f\n", params->scale);
        fprintf(file, "offset=%f\n", params->offset);
    }

    fclose(file);
    return ACTIVATION_SUCCESS;
}

/**
 * @brief Load activation function configuration
 */
int activation_save_config(const char* filename, ActivationType type, 
                          const ActivationParams* params) {
    if (!filename) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }
    
    fprintf(file, "activation_type=%d\n", type);
    if (params) {
        fprintf(file, "alpha=%f\n", params->alpha);
        fprintf(file, "beta=%f\n", params->beta);
        fprintf(file, "gamma=%f\n", params->gamma);
        fprintf(file, "lambda=%f\n", params->lambda);
        fprintf(file, "threshold=%f\n", params->threshold);
        fprintf(file, "scale=%f\n", params->scale);
        fprintf(file, "offset=%f\n", params->offset);
    }
    
    fclose(file);
    return ACTIVATION_SUCCESS;
}

/**
 * @brief Load activation function configuration
 */
void activation_example_usage(void) {
    printf("Activation Functions Library Example\n");
    printf("===================================\n\n");

    // Test basic activation functions
    double test_input = 0.5;
    ActivationParams params;

    printf("Testing with input value: %.2f\n\n", test_input);

    // Test sigmoid
    double sigmoid_result = activation_sigmoid(test_input, NULL);
    printf("Sigmoid: %.6f\n", sigmoid_result);

    // Test ReLU
    double relu_result = activation_relu(test_input, NULL);
    printf("ReLU: %.6f\n", relu_result);

    // Test Leaky ReLU with custom alpha
    activation_init_params(&params, ACTIVATION_LEAKY_RELU);
    params.alpha = 0.1;
    double leaky_relu_result = activation_leaky_relu(test_input, &params);
    printf("Leaky ReLU (alpha=0.1): %.6f\n", leaky_relu_result);

    // Test vectorized operations
    double input_vector[] ={-2.0, -1.0, 0.0, 1.0, 2.0};
    double output_vector[5];
    int size = 5;

    printf("\nVectorized ReLU:\n");
    printf("Input:  ");
    for (int i = 0; i < size; i++) printf("%.1f ", input_vector[i]);
    printf("\n");

    activation_apply_vector(ACTIVATION_RELU, input_vector, output_vector, size, NULL);
    printf("Output: ");
    for (int i = 0; i < size; i++) printf("%.1f ", output_vector[i]);
    printf("\n\n");

    // Test softmax
    printf("Softmax example:\n");
    double softmax_input[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double softmax_output[5];
    activation_softmax(softmax_input, softmax_output, 5, NULL);

    printf("Input: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", softmax_input[i]);
    printf("\n");
    printf("Output: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", softmax_output[i]);
    printf("\n");

    // Verify softmax sums to 1
    double sum = 0.0;
    for (int i = 0; i < 5; i++) sum += softmax_output[i];
    printf("Sum: %.6f (should be 1.0)\n", sum);
}

/* =============================================================================
 * ADVANCED VECTORIZED OPERATIONS WITH SIMD SUPPORT
 * ============================================================================= */

#ifdef __AVX2__

int activation_relu_avx2(const double* input, double* output, int size) {
    if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    const int simd_size = 4;
    const int simd_iterations = size / simd_size;
    const int remainder = size % simd_size;

    __m256d zero = _mm256_setzero_pd();

    for (int i = 0; i < simd_iterations; i++) {
        __m256d input_vec = _mm256_loadu_pd(&input[i * simd_size]);
        __m256d result = _mm256_max_pd(input_vec, zero);
        _mm256_storeu_pd(&output[i * simd_size], result);
    }

    for (int i = simd_iterations * simd_size; i < size; i++) {
        output[i] = activation_relu(input[i], NULL);
    }

    return ACTIVATION_SUCCESS;
}

int activation_sigmoid_avx2(const double* input, double* output, int size) {
    if (!input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    const int simd_size = 4;
    const int simd_iterations = size / simd_size;

    __m256d ones = _mm256_set1_pd(1.0);
    __m256d clip_val = _mm256_set1_pd(ACTIVATION_CLIP_VALUE);
    __m256d neg_clip_val = _mm256_set1_pd(-ACTIVATION_CLIP_VALUE);

    for (int i = 0; i < simd_iterations; i++) {
        __m256d x = _mm256_loadu_pd(&input[i * simd_size]);
        x = _mm256_max_pd(x, neg_clip_val);
        x = _mm256_min_pd(x, clip_val);
        x = _mm256_sub_pd(_mm256_setzero_pd(), x);

        __m256d exp_x = _mm256_set_pd(exp(x[3]), exp(x[2]), exp(x[1]), exp(x[0]));
        __m256d result = _mm256_div_pd(ones, _mm256_add_pd(ones, exp_x));
        _mm256_storeu_pd(&output[i * simd_size], result);
    }

    for (int i = simd_iterations * simd_size; i < size; i++) {
        output[i] = activation_sigmoid(input[i], NULL);
    }

    return ACTIVATION_SUCCESS;
}
#endif

/* =============================================================================
 * BATCH PROCESSING AND PARALLEL EXECUTION
 * ============================================================================= */

typedef struct {
    ActivationType type;
    const double* input;
    double* output;
    int start_idx;
    int end_idx;
    const ActivationParams* params;
} ThreadData;

#ifdef _OPENMP

int activation_apply_parallel(ActivationType type, const double* input, double* output,
                             int size, const ActivationParams* params, int num_threads) {
    if (!input || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    if (num_threads <= 0) num_threads = omp_get_max_threads();

    #pragma omg parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        output[i] = activation_apply(type, input[i], params);
    }

    return ACTIVATION_SUCCESS;
}

int activation_batch_process(ActivationType* type, const double** inputs, double** outputs,
                            int* size, int batch_count, const ActivationParams* params) {
    if (!types || !inputs || !outputs || !sizes || batch_count <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    #pragma omp parallel for
    for (int b = 0; b < batch_count; b++) {
        activation_apply_vector(types[b], inputs[b], outputs[b], sizes[b], params);
    }

    return ACTIVATION_SUCCESS;
}
#endif

/* =============================================================================
 * MEMORY POOL MANAGEMENT
 * ============================================================================= */

typedef struct MemoryBlock {
    void* data;
    size_t size;
    int is_free;
    struct MemoryBlock* next;
} MemoryBlock;

typedef struct {
    MemoryBlock* blocks;
    size_t total_size;
    size_t used_size;
    int block_count;
} MemoryPool;

static MemoryPool global_pool = {NULL, 0, 0, 0};

int activation_init_memory_pool(size_t pool_size) {
    if (global_pool.blocks) {
        return ACTIVATION_ERROR_PARAM;
    }

    global_pool.blocks = malloc(sizeof(MemoryBlock));
    if (!global_pool.blocks) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    global_pool.blocks->data = malloc(pool_size);
    if (!global_pool.blocks->data) {
        free(global_pool.blocks);
        global_pool.blocks = NULL;
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }
    
    global_pool.blocks->size = pool_size;
    global_pool.blocks->is_free = 1;
    global_pool.blocks->next = NULL;
    global_pool.total_size = pool_size;
    global_pool.used_size = 0;
    global_pool.block_count = 1;

    return ACTIVATION_SUCCESS;
}

void* activation_alloc_from_pool(size_t size) {
    if (!global_pool.blocks) return NULL;

    size = (size + 7) & ~7;

    MemoryBlock* current = global_pool.blocks;
    while (current) {
        if (current->is_free && current->size >= size) {
            if (current->size > size + sizeof(MemoryBlock)) {
                MemoryBlock* new_block = (MemoryBlock*)((char*)current->data + size);
                new_block->data = (char*)current->data + size + sizeof(MemoryBlock);
                new_block->size = current->size - size - sizeof(MemoryBlock);
                new_block->is_free = 1;
                new_block->next = current->next;
                current->next = new_block;
                global_pool.block_count++;
            }
            current->is_free = 0;
            current->size = size;
            global_pool.used_size += size;
            return current->data;
        }
        current = current->next;
    }

    return NULL;
}

void activation_free_to_pool(void* ptr) {
    if (!ptr || !global_pool.blocks) return;

    MemoryBlock* current = global_pool.blocks;
    while (current) {
        if (current->data == ptr) {
            current->is_free = 1;
            global_pool.used_size -= current->size;
            break;
        }
        current = current->next;
    }
}

void activation_cleanup_memory_pool(void) {
    if (!global_pool.blocks) return;

    MemoryBlock* current = global_pool.blocks;
    while (current) {
        MemoryBlock* next = current->next;
        if (current == global_pool.blocks) {
            free(current->data);
        }
        free(current);
        current = next;
    }

    global_pool.blocks = NULL;
    global_pool.total_size = 0;
    global_pool.used_size = 0;
    global_pool.block_count = 0;
}

/* =============================================================================
 * ACTIVATION LAYER IMPLEMENTATION
 * ============================================================================= */

typedef struct {
    ActivationType type;
    ActivationParams params;
    double* input_buffer;
    double* output_buffer;
    double* gradient_buffer;
    int size;
    int is_training;
    double dropout_rate;
    char* mask_buffer;
} ActivationLayer;

ActivationLayer* activation_layer_create(ActivationType type, int size,
                                        const ActivationParams* params) {
    if (size <= 0) {
        report_error(ACTIVATION_ERROR_SIZE);
        return NULL;
    }

    ActivationLayer* layer = malloc(sizeof(ActivationLayer));
    if (!layer) {
        report_error(ACTIVATION_ERROR_NULL);
        return NULL;
    }

    layer->type = type;
    layer->size = size;
    layer->is_training = 0;
    layer->dropout_rate = 0.0;

    if (params) {
        layer->params = *params;
    } else {
        activation_init_params(&layer->params, type);
    }

    layer->input_buffer = activation_alloc_from_pool(size * sizeof(double));
    layer->output_buffer = activation_alloc_from_pool(size * sizeof(double));
    layer->gradient_buffer = activation_alloc_from_pool(size * sizeof(double));
    layer->mask_buffer = activation_alloc_from_pool(size * sizeof(char));

    if (!layer->input_buffer || !layer->output_buffer ||
        !layer->gradient_buffer || !layer->mask_buffer) {
        activation_layer_destroy(layer);
        report_error(ACTIVATION_ERROR_NULL);
        return NULL;
    }

    return layer;
}

void activation_layer_destroy(ActivationLayer* layer) {
    if (!layer) return;

    if (layer->input_buffer) activation_free_to_pool(layer->input_buffer);
    if (layer->output_buffer) activation_free_to_pool(layer->output_buffer);
    if (layer->gradient_buffer) activation_free_to_pool(layer->gradient_buffer);
    if (layer->mask_buffer) activation_free_to_pool(layer->mask_buffer);

    free(layer);
}

int activation_layer_forward(ActivationLayer* layer, const double* input, double* output) {
    if (!layer || !input || !output) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    memcpy(layer->input_buffer, input, layer->size * sizeof(double));

    int result = activation_apply_vector(layer->type, layer->input_buffer,
                                        layer->output_buffer, layer->size, &layer->params);
    if (result != ACTIVATION_SUCCESS) return result;

    if (layer->is_training && layer->dropout_rate > 0.0) {
        for (int i = 0; i < layer->size; i++) {
            double rand_val = (double)rand() / RAND_MAX;
            if (rand_val < layer->dropout_rate) {
                layer->mask_buffer[i] = 0;
                layer->output_buffer[i] = 0.0;
            } else {
                layer->mask_buffer[i] = 1;
                layer->output_buffer[i] /= (1.0 - layer->dropout_rate);
            }
        }
    }

    memcpy(output, layer->output_buffer, layer->size * sizeof(double));
    return ACTIVATION_SUCCESS;
}

int activation_layer_backward(ActivationLayer* layer, const double* grad_output,
                             double* grad_input) {
    if (!layer || !grad_input || !grad_input) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    if (layer->type == ACTIVATION_SOFTMAX || layer->type == ACTIVATION_LOG_SOFTMAX) {
        for (int i = 0; i < layer->size; i++) {
            layer->gradient_buffer[i] = grad_output[i];
        }
    } else {
        int result = activation_apply_derivative_vector(layer->type, layer->output_buffer,
                                                        layer->gradient_buffer, layer->size,
                                                        &layer->params);
        if (result != ACTIVATION_SUCCESS) return result;

        for (int i = 0; i < layer->size; i++) {
            layer->gradient_buffer[i] *= grad_output[i];
        }
    }

    if (layer->is_training && layer->dropout_rate > 0.0) {
        for (int i = 0; i < layer->size; i++) {
            if (layer->mask_buffer[i] == 0) {
                layer->gradient_buffer[i] = 0.0;
            } else {
                layer->gradient_buffer[i] /= (1.0 - layer->dropout_rate);
            }
        }
    }

    memcpy(grad_input, layer->gradient_buffer, layer->size * sizeof(double));
    return ACTIVATION_SUCCESS;
}

void activation_layer_set_training(ActivationLayer* layer, int is_training) {
    if (layer) {
        layer->is_training = is_training;
    }
}

void activation_layer_set_dropout(ActivationLayer* layer, double dropout_rate) {
    if (layer && dropout_rate >= 0.0 && dropout_rate < 1.0) {
        layer->dropout_rate = dropout_rate;
    }
}

/* =============================================================================
 * ACTIVATION FUNCTION FUSION
 * ============================================================================= */

typedef struct {
    ActivationType primary_type;
    ActivationType secondary_type;
    double blend_factor;
    int use_gating;
    ActivationParams primary_params;
    ActivationParams secondary_params;
} FusedActivation;

double activation_fused_apply(const FusedActivation* fused, double x) {
    if (!fused) return 0.0;

    double primary_result = activation_apply(fused->primary_type, x, &fused->primary_params);
    double secondary_result = activation_apply(fused->secondary_type, x, &fused->secondary_params);

    if (fused->use_gating) {
        double gate = activation_sigmoid(x, NULL);
        return gate * primary_result + (1.0 - gate) * secondary_result;
    } else {
        return fused->blend_factor * primary_result + (1.0 - fused->blend_factor) * secondary_result;
    }
}

int activation_fused_vector(const FusedActivation* fused, const double* input, 
                           double* output, int size) {
    if (!fused || !input || !output || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    for (int i = 0; i < size; i++) {
        output[i] = activation_fused_apply(fused, input[i]);
    }

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * ADAPTIVE ACTIVATION FUNCTIONS
 * ============================================================================= */

typedef struct {
    ActivationType base_type;
    double* adaptive_params;
    int param_count;
    double learning_rate;
    double momentum;
    double* param_gradients;
    double* param_velocity;
} AdaptiveActivation;

AdaptiveActivation* activation_adaptive_create(ActivationType base_type, int param_count,
                                              double learning_rate) {
    if (param_count <= 0 || learning_rate <= 0.0) {
        report_error(ACTIVATION_ERROR_PARAM);
        return NULL;
    }

    AdaptiveActivation* adaptive = malloc(sizeof(AdaptiveActivation));
    if (!adaptive) {
        report_error(ACTIVATION_ERROR_NULL);
        return NULL;
    }

    adaptive->base_type = base_type;
    adaptive->param_count = param_count;
    adaptive->learning_rate = learning_rate;
    adaptive->momentum = 0.9;

    adaptive->adaptive_params = calloc(param_count, sizeof(double));
    adaptive->param_gradients = calloc(param_count, sizeof(double));
    adaptive->param_velocity = calloc(param_count, sizeof(double));

    if (!adaptive->adaptive_params || !adaptive->param_gradients || !adaptive->param_velocity) {
        activation_adaptive_destroy(adaptive);
        report_error(ACTIVATION_ERROR_NULL);
        return NULL;
    }

    for (int i = 0; i < param_count; i++) {
        adaptive->adaptive_params[i] = 0.01 * ((double)rand() / RAND_MAX - 0.5);
    }

    return adaptive;
}

void activation_adaptive_destroy(AdaptiveActivation* adaptive) {
    if (!adaptive) return;

    free(adaptive->adaptive_params);
    free(adaptive->param_gradients);
    free(adaptive->param_velocity);
    free(adaptive);
}

double activation_adaptive_forward(AdaptiveActivation* adaptive, double x) {
    if (!adaptive) return 0.0;

    double base_result = activation_apply(adaptive->base_type, x, NULL);
    double adaptation = 0.0;

    for (int i = 0; i < adaptive->param_count; i++) {
        adaptation += adaptive->adaptive_params[i] * pow(x, i);
    }

    return base_result + adaptation;
}

void activation_adaptive_backward(AdaptiveActivation* adaptive, double x, double grad_output) {
    if (!adaptive) return;

    for (int i = 0; i < adaptive->param_count; i++) {
        adaptive->param_gradients[i] += grad_output * pow(x, i);
    }
}

void activation_adaptive_update(AdaptiveActivation* adaptive) {
    if (!adaptive) return;

    for (int i = 0; i < adaptive->param_count; i++) {
        adaptive->param_velocity[i] = adaptive->momentum * adaptive->param_velocity[i] +
                                     adaptive->learning_rate * adaptive->param_gradients[i];
        adaptive->adaptive_params[i] -= adaptive->param_velocity[i];
        adaptive->param_gradients[i] = 0.0;
    }
}

/* =============================================================================
 * ACTIVATION FUNCTION OPTIMIZATION
 * ============================================================================= */

typedef struct {
    ActivationType type;
    double* param_values;
    int param_count;
    double cost;
    double* gradient;
} ActivationOptimization;

double activation_objective_function(const double* params, int param_count, 
                                   const double* input, const double* target, 
                                   int size, ActivationType type) {
    ActivationParams activation_params;
    activation_init_params(&activation_params, type);

    if (param_count > 0) activation_params.alpha = params[0];
    if (param_count > 1) activation_params.beta = params[1];
    if (param_count > 2) activation_params.gamma = params[2];

    double total_error = 0.0;
    for (int i = 0; i < size; i++) {
        double output = activation_apply(type, input[i], &activation_params);
        double error = output - target[i];
        total_error += error * error;
    }

    return total_error / size;
}

int activation_optimization_parameters(ActivationType type, const double* input,
                                     const double* target, int size,
                                     double* optimal_params, int param_count,
                                     int max_iterations, double tolerance) {
    if (!input || !target || !optimal_params || size <= 0 || param_count <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    double* current_params = malloc(param_count * sizeof(double));
    double* gradient = malloc(param_count * sizeof(double));
    double* best_params = malloc(param_count * sizeof(double));

    if (!current_params || !gradient || !best_params) {
        free(current_params);
        free(gradient);
        free(best_params);
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    for (int i = 0; i < param_count; i++) {
        current_params[i] = optimal_params[i];
        best_params[i] = optimal_params[i];
    }

    double best_cost = activation_objective_function(current_params, param_count,
                                                    input, target, size, type);
    double learning_rate = 0.01;
    double epsilon = 1e-8;

    for (int iter = 0; iter < max_iterations; iter++) {
        for (int p = 0; p < param_count; p++) {
            double original_param = current_params[p];

            current_params[p] = original_param + epsilon;
            double cost_plus = activation_objective_function(current_params, param_count,
                                                           input, target, size, type);

            current_params[p] = original_param - epsilon;
            double cost_minus = activation_objective_function(current_params, param_count,
                                                            input, target, size, type);

            gradient[p] = (cost_plus - cost_minus) / (2.0 * epsilon);
            current_params[p] = original_param;
        }

        for (int p = 0; p < param_count; p++) {
            current_params[p] -= learning_rate * gradient[p];
        }

        double current_cost = activation_objective_function(current_params, param_count,
                                                          input, target, size, type);

        if (current_cost < best_cost) {
            best_cost = current_cost;
            memcpy(best_params, current_params, param_count * sizeof(double));
        }

        if (best_cost < tolerance) break;
    }

    memcpy(optimal_params, best_params, param_count * sizeof(double));

    free(current_params);
    free(gradient);
    free(best_params);

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * ACTIVATION FUNCTION ANALYSIS
 * ============================================================================= */

typedef struct {
    double saturation_point;
    double dead_neuron_ratio;
    double gradient_variance;
    double output_variance;
    double lipschitz_constant;
} ActivationAnalysis;

int activation_analyze_function(ActivationType type, const ActivationParams* params,
                              const double* input, int size, ActivationAnalysis* analysis) {
    if (!input || !analysis || size <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    double* outputs = malloc(size * sizeof(double));
    double* gradients = malloc(size * sizeof(double));

    if (!outputs || !gradients) {
        free(outputs);
        free(gradients);
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    activation_apply_vector(type, input, outputs, size, params);
    activation_apply_derivative_vector(type, outputs, gradients, size, params);

    int dead_neurons = 0;
    double output_sum = 0.0, output_sq_sum = 0.0;
    double gradient_sum = 0.0, gradient_sq_sum = 0.0;
    double max_gradient = 0.0;

    for (int i = 0; i < size; i++) {
        if (fabs(gradients[i]) < 1e-6) dead_neurons++;

        output_sum += outputs[i];
        output_sq_sum += outputs[i] * outputs[i];
        gradient_sum += gradients[i];
        gradient_sq_sum += gradients[i] * gradients[i];

        if (fabs(gradients[i]) > max_gradient) {
            max_gradient = fabs(gradients[i]);
        }
    }

    analysis->dead_neuron_ratio = (double)dead_neurons / size;
    analysis->output_variance = (output_sq_sum / size) - pow(output_sum / size, 2);
    analysis->gradient_variance = (gradient_sq_sum / size) - pow(gradient_sum / size, 2);
    analysis->lipschitz_constant = max_gradient;

    int saturated_neurons = 0;
    const ActivationInfo* info = activation_get_info(type);
    if (info && info->is_bounded) {
        for (int i = 0; i < size; i++) {
            if (fabs(outputs[i] - info->range_max) < 1e-3 ||
                fabs(outputs[i] - info->range_min) < 1e-3) {
                saturated_neurons++;
            }
        }
    }
    analysis->saturation_point = (double)saturated_neurons / size;

    free(outputs);
    free(gradients);

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * ACTIVATION FUNCTION VISUALIZATION DATA
 * ============================================================================= */

int activation_generate_plot_data(ActivationType type, const ActivationParams* params,
                                double x_min, double x_max, int num_points,
                                double* x_values, double* y_values, double* dy_values) {
    if (!x_values || !y_values || num_points <= 0) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    double step = (x_max - x_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        x_values[i] = x_min + i * step;
        y_values[i] = activation_apply(type, x_values[i], params);
        
        if (dy_values) {
            dy_values[i] = activation_apply_derivative(type, y_values[i], params);
        }
    }

    return ACTIVATION_SUCCESS;
}

/* =============================================================================
 * ACTIVATION FUNCTION SERIALIZATION
 * ============================================================================= */

int activation_serialize_config(const char* filename, ActivationType type,
                               const ActivationParams* params, const char* metadata) {
    if (!filename) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    FILE* file = fopen(filename, "wb");
    if (!file) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    uint32_t magic_number = 0xACF12345;
    uint32_t version = 1;

    fwrite(&magic_number, sizeof(uint32_t), 1, file);
    fwrite(&version, sizeof(uint32_t), 1, file);
    fwrite(&type, sizeof(ActivationType), 1, file);

    if (params) {
        uint8_t has_params = 1;
        fwrite(&has_params, sizeof(uint8_t), 1, file);
        fwrite(params, sizeof(ActivationParams), 1, file);
    } else {
        uint8_t has_params = 0;
        fwrite(&has_params, sizeof(uint8_t), 1, file);
    }

    if (metadata) {
        uint32_t metadata_len = strlen(metadata) + 1;
        fwrite(&metadata_len, sizeof(uint32_t), 1, file);
        fwrite(metadata, metadata_len, 1, file);
    } else {
        uint32_t metadata_len = 0;
        fwrite(&metadata_len, sizeof(uint32_t), 1, file);
    }

    fclose(file);
    return ACTIVATION_SUCCESS;
}

int activation_deserialize_config(const char* filename, ActivationType* type,
                                 ActivationParams* params, char* metadata, int metadata_size) {
    if (!filename || !type) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    FILE* file = fopen(filename, "rb");
    if (!file) {
        report_error(ACTIVATION_ERROR_NULL);
        return ACTIVATION_ERROR_NULL;
    }

    uint32_t magic_number, version;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&version, sizeof(uint32_t), 1, file);
    
    if (magic_number != 0xACF12345 || version != 1) {
        fclose(file);
        report_error(ACTIVATION_ERROR_PARAM);
        return ACTIVATION_ERROR_PARAM;
    }
    
    fread(type, sizeof(ActivationType), 1, file);
    
    uint8_t has_params;
    fread(&has_params, sizeof(uint8_t), 1, file);
    
    if (has_params && params) {
        fread(params, sizeof(ActivationParams), 1, file);
    } else if (has_params) {
        fseek(file, sizeof(ActivationParams), SEEK_CUR);
    }
    
    uint32_t metadata_len;
    fread(&metadata_len, sizeof(uint32_t), 1, file);
    
    if (metadata_len > 0 && metadata && metadata_size > 0) {
        int read_len = (metadata_len < metadata_size) ? metadata_len : metadata_size - 1;
        fread(metadata, read_len, 1, file);
        metadata[read_len] = '\0';
    }

    fclose(file);
    return ACTIVATION_SUCCESS;
}