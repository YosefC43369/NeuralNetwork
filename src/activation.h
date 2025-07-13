#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * =============================================================================
 * ACTIVATION FUNCTIONS HEADER FILE
 * =============================================================================
 * 
 * This header file contains comprehensive activation function declarations
 * for neural network implementations. It provides various activation functions
 * commonly used in machine learning and deep learning applications.
 * 
 * Features:
 * - Multiple activation function types
 * - Vectorized operations support
 * - Derivative calculations
 * - In-place and out-of-place operations
 * - Parameter validation
 * - Thread-safe implementations
 * 
 * Author: Neural Network Project
 * Version: 2.0
 * Date: 2025
 * 
 * =============================================================================
 */

/* =============================================================================
 * CONSTANTS AND MACROS
 * ============================================================================= */

/* Mathematical constants */
#define ACTIVATION_E            2.71828182845904523536
#define ACTIVATION_PI           3.14159265358979323846
#define ACTIVATION_SQRT_2_PI    2.50662827463100050242
#define ACTIVATION_EPSILON      1e-15   /* Small value to prevent division by zero */
#define ACTIVATION_CLIP_VALUE   50.0    /* Clipping value for numerical stability */

/* Error codes */
#define ACTIVATION_SUCCESS      0
#define ACTIVATION_ERROR_NULL   -1
#define ACTIVATION_ERROR_SIZE   -2
#define ACTIVATION_ERROR_PARAM  -3
#define ACTIVATION_ERROR_RANGE  -4

/* Maximum supported vector size for vectorized operations */
#define MAX_VECTOR_SIZE         10000

/* =============================================================================
 * ACTIVATION FUNCTION ENUMERATION
 * ============================================================================= */

/**
 * @brief Enumeration of supported activation function types
 * 
 * This enum defines all the activation functions supported by the library.
 * Each function has both forward and derivative implementations.
 */
typedef enum {
    /* Basic activation functions */
    ACTIVATION_SIGMOID = 0,         /**< Sigmoid: f(x) = 1/(1+e^(-x)) */
    ACTIVATION_TANH,                /**< Hyperbolic tangent: f(x) = tanh(x) */
    ACTIVATION_RELU,                /**< ReLU: f(x) = max(0,x) */
    ACTIVATION_LEAKY_RELU,          /**< Leaky ReLU: f(x) = max(αx,x) */
    
    /* Advanced ReLU variants */
    ACTIVATION_ELU,                 /**< ELU: f(x) = x if x>0, α(e^x-1) if x≤0 */
    ACTIVATION_SELU,                /**< SELU: Self-normalizing ELU */
    ACTIVATION_SWISH,               /**< Swish: f(x) = x * sigmoid(x) */
    ACTIVATION_MISH,                /**< Mish: f(x) = x * tanh(softplus(x)) */
    ACTIVATION_GELU,                /**< GELU: Gaussian Error Linear Unit */
    
    /* Step and linear functions */
    ACTIVATION_STEP,                /**< Step function: f(x) = 1 if x>0, 0 if x≤0 */
    ACTIVATION_LINEAR,              /**< Linear: f(x) = x */
    ACTIVATION_SOFTPLUS,            /**< Softplus: f(x) = ln(1+e^x) */
    ACTIVATION_SOFTSIGN,            /**< Softsign: f(x) = x/(1+|x|) */
    
    /* Exponential and logarithmic */
    ACTIVATION_EXPONENTIAL,         /**< Exponential: f(x) = e^x */
    ACTIVATION_HARD_SIGMOID,        /**< Hard sigmoid: fast approximation */
    ACTIVATION_HARD_TANH,           /**< Hard tanh: fast approximation */
    
    /* Probabilistic functions */
    ACTIVATION_SOFTMAX,             /**< Softmax: for multi-class classification */
    ACTIVATION_LOG_SOFTMAX,         /**< Log softmax: numerically stable */
    
    /* Custom and experimental */
    ACTIVATION_CUSTOM,              /**< User-defined custom function */
    ACTIVATION_COUNT                /**< Total number of activation types */
} ActivationType;

/* =============================================================================
 * ACTIVATION FUNCTION PARAMETERS STRUCTURE
 * ============================================================================= */

/**
 * @brief Structure to hold parameters for parameterized activation functions
 * 
 * Some activation functions require additional parameters (e.g., alpha for Leaky ReLU).
 * This structure holds all possible parameters that might be needed.
 */
typedef struct {
    double alpha;           /**< Alpha parameter (e.g., for Leaky ReLU, ELU) */
    double beta;            /**< Beta parameter (e.g., for Swish, custom functions) */
    double gamma;           /**< Gamma parameter (for advanced functions) */
    double lambda;          /**< Lambda parameter (for SELU) */
    double threshold;       /**< Threshold parameter (for thresholded functions) */
    double scale;           /**< Scale parameter (for normalized functions) */
    double offset;          /**< Offset parameter (for shifted functions) */
    
    /* Custom function pointers */
    double (*custom_func)(double x, void* params);      /**< Custom activation function */
    double (*custom_deriv)(double x, void* params);     /**< Custom derivative function */
    void* custom_params;                                /**< Custom parameters */
} ActivationParams;

/* =============================================================================
 * ACTIVATION FUNCTION INFORMATION STRUCTURE
 * ============================================================================= */

/**
 * @brief Structure containing information about an activation function
 * 
 * This structure provides metadata about activation functions including
 * their properties, recommended use cases, and parameter requirements.
 */
typedef struct {
    ActivationType type;            /**< Function type identifier */
    const char* name;               /**< Human-readable name */
    const char* formula;            /**< Mathematical formula */
    const char* description;        /**< Detailed description */
    double range_min;               /**< Minimum output value */
    double range_max;               /**< Maximum output value */
    int is_bounded;                 /**< Whether output is bounded */
    int is_monotonic;               /**< Whether function is monotonic */
    int is_differentiable;          /**< Whether function is differentiable */
    int has_vanishing_gradient;     /**< Whether prone to vanishing gradients */
    int requires_params;            /**< Whether requires parameters */
    const char* recommended_use;    /**< Recommended use cases */
} ActivationInfo;

/* =============================================================================
 * FUNCTION POINTER TYPES
 * ============================================================================= */

/**
 * @brief Function pointer type for activation functions
 * @param x Input value
 * @param params Optional parameters
 * @return Activated output value
 */
typedef double (*ActivationFunc)(double x, const ActivationParams* params);

/**
 * @brief Function pointer type for activation derivative functions
 * @param x Input value (or activated output for some functions)
 * @param params Optional parameters
 * @return Derivative value
 */
typedef double (*ActivationDerivFunc)(double x, const ActivationParams* params);

/**
 * @brief Function pointer type for vectorized activation functions
 * @param input Input array
 * @param output Output array
 * @param size Array size
 * @param params Optional parameters
 * @return Error code
 */
typedef int (*ActivationVectorFunc)(const double* input, double* output, 
                                   int size, const ActivationParams* params);

/* =============================================================================
 * BASIC ACTIVATION FUNCTIONS
 * ============================================================================= */

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return Sigmoid output: 1/(1+e^(-x))
 */
double activation_sigmoid(double x, const ActivationParams* params);

/**
 * @brief Derivative of sigmoid function
 * @param x Activated output value (sigmoid(x))
 * @param params Optional parameters (can be NULL)
 * @return Derivative: x * (1 - x)
 */
double activation_sigmoid_derivative(double x, const ActivationParams* params);

/**
 * @brief Hyperbolic tangent activation function
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return Tanh output: tanh(x)
 */
double activation_tanh(double x, const ActivationParams* params);

/**
 * @brief Derivative of hyperbolic tangent function
 * @param x Activated output value (tanh(x))
 * @param params Optional parameters (can be NULL)
 * @return Derivative: 1 - x^2
 */
double activation_tanh_derivative(double x, const ActivationParams* params);

/**
 * @brief ReLU (Rectified Linear Unit) activation function
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return ReLU output: max(0, x)
 */
double activation_relu(double x, const ActivationParams* params);

/**
 * @brief Derivative of ReLU function
 * @param x Input value (not activated output)
 * @param params Optional parameters (can be NULL)
 * @return Derivative: 1 if x > 0, 0 otherwise
 */
double activation_relu_derivative(double x, const ActivationParams* params);

/**
 * @brief Leaky ReLU activation function
 * @param x Input value
 * @param params Parameters containing alpha value (default: 0.01)
 * @return Leaky ReLU output: max(alpha*x, x)
 */
double activation_leaky_relu(double x, const ActivationParams* params);

/**
 * @brief Derivative of Leaky ReLU function
 * @param x Input value (not activated output)
 * @param params Parameters containing alpha value
 * @return Derivative: 1 if x > 0, alpha otherwise
 */
double activation_leaky_relu_derivative(double x, const ActivationParams* params);

/* =============================================================================
 * ADVANCED ACTIVATION FUNCTIONS
 * ============================================================================= */

/**
 * @brief ELU (Exponential Linear Unit) activation function
 * @param x Input value
 * @param params Parameters containing alpha value (default: 1.0)
 * @return ELU output: x if x > 0, alpha*(e^x - 1) if x <= 0
 */
double activation_elu(double x, const ActivationParams* params);

/**
 * @brief Derivative of ELU function
 * @param x Input value
 * @param params Parameters containing alpha value
 * @return Derivative: 1 if x > 0, alpha*e^x if x <= 0
 */
double activation_elu_derivative(double x, const ActivationParams* params);

/**
 * @brief SELU (Scaled Exponential Linear Unit) activation function
 * @param x Input value
 * @param params Optional parameters (uses standard SELU constants)
 * @return SELU output: scale * ELU(x, alpha)
 */
double activation_selu(double x, const ActivationParams* params);

/**
 * @brief Derivative of SELU function
 * @param x Input value
 * @param params Optional parameters
 * @return SELU derivative
 */
double activation_selu_derivative(double x, const ActivationParams* params);

/**
 * @brief Swish activation function
 * @param x Input value
 * @param params Parameters containing beta value (default: 1.0)
 * @return Swish output: x * sigmoid(beta * x)
 */
double activation_swish(double x, const ActivationParams* params);

/**
 * @brief Derivative of Swish function
 * @param x Input value
 * @param params Parameters containing beta value
 * @return Swish derivative
 */
double activation_swish_derivative(double x, const ActivationParams* params);

/**
 * @brief Mish activation function
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return Mish output: x * tanh(softplus(x))
 */
double activation_mish(double x, const ActivationParams* params);

/**
 * @brief Derivative of Mish function
 * @param x Input value
 * @param params Optional parameters
 * @return Mish derivative
 */
double activation_mish_derivative(double x, const ActivationParams* params);

/**
 * @brief GELU (Gaussian Error Linear Unit) activation function
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return GELU output: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
double activation_gelu(double x, const ActivationParams* params);

/**
 * @brief Derivative of GELU function
 * @param x Input value
 * @param params Optional parameters
 * @return GELU derivative
 */
double activation_gelu_derivative(double x, const ActivationParams* params);

/* =============================================================================
 * UTILITY ACTIVATION FUNCTIONS
 * ============================================================================= */

/**
 * @brief Step activation function
 * @param x Input value
 * @param params Parameters containing threshold (default: 0.0)
 * @return Step output: 1 if x > threshold, 0 otherwise
 */
double activation_step(double x, const ActivationParams* params);

/**
 * @brief Linear activation function
 * @param x Input value
 * @param params Parameters containing scale and offset (default: scale=1, offset=0)
 * @return Linear output: scale * x + offset
 */
double activation_linear(double x, const ActivationParams* params);

/**
 * @brief Derivative of linear function
 * @param x Input value
 * @param params Parameters containing scale
 * @return Linear derivative: scale
 */
double activation_linear_derivative(double x, const ActivationParams* params);

/**
 * @brief Softplus activation function
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return Softplus output: ln(1 + e^x)
 */
double activation_softplus(double x, const ActivationParams* params);

/**
 * @brief Derivative of Softplus function
 * @param x Input value
 * @param params Optional parameters
 * @return Softplus derivative: sigmoid(x)
 */
double activation_softplus_derivative(double x, const ActivationParams* params);

/**
 * @brief Softsign activation function
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return Softsign output: x / (1 + |x|)
 */
double activation_softsign(double x, const ActivationParams* params);

/**
 * @brief Derivative of Softsign function
 * @param x Input value
 * @param params Optional parameters
 * @return Softsign derivative: 1 / (1 + |x|)^2
 */
double activation_softsign_derivative(double x, const ActivationParams* params);

/* =============================================================================
 * HARD APPROXIMATION FUNCTIONS
 * ============================================================================= */

/**
 * @brief Hard sigmoid activation function (fast approximation)
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return Hard sigmoid output: max(0, min(1, 0.2*x + 0.5))
 */
double activation_hard_sigmoid(double x, const ActivationParams* params);

/**
 * @brief Derivative of hard sigmoid function
 * @param x Input value
 * @param params Optional parameters
 * @return Hard sigmoid derivative: 0.2 if -2.5 < x < 2.5, 0 otherwise
 */
double activation_hard_sigmoid_derivative(double x, const ActivationParams* params);

/**
 * @brief Hard tanh activation function (fast approximation)
 * @param x Input value
 * @param params Optional parameters (can be NULL)
 * @return Hard tanh output: max(-1, min(1, x))
 */
double activation_hard_tanh(double x, const ActivationParams* params);

/**
 * @brief Derivative of hard tanh function
 * @param x Input value
 * @param params Optional parameters
 * @return Hard tanh derivative: 1 if -1 < x < 1, 0 otherwise
 */
double activation_hard_tanh_derivative(double x, const ActivationParams* params);

/* =============================================================================
 * SOFTMAX AND LOG-SOFTMAX FUNCTIONS
 * ============================================================================= */

/**
 * @brief Softmax activation function (for vectors)
 * @param input Input vector
 * @param output Output vector (must be pre-allocated)
 * @param size Vector size
 * @param params Optional parameters (can be NULL)
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_softmax(const double* input, double* output, int size, 
                      const ActivationParams* params);

/**
 * @brief Log-softmax activation function (numerically stable)
 * @param input Input vector
 * @param output Output vector (must be pre-allocated)
 * @param size Vector size
 * @param params Optional parameters (can be NULL)
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_log_softmax(const double* input, double* output, int size, 
                          const ActivationParams* params);

/* =============================================================================
 * VECTORIZED OPERATIONS
 * ============================================================================= */

/**
 * @brief Apply activation function to entire vector
 * @param type Activation function type
 * @param input Input vector
 * @param output Output vector (can be same as input for in-place operation)
 * @param size Vector size
 * @param params Optional parameters
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_apply_vector(ActivationType type, const double* input, 
                           double* output, int size, const ActivationParams* params);

/**
 * @brief Apply activation derivative to entire vector
 * @param type Activation function type
 * @param input Input vector (or activated output for some functions)
 * @param output Output vector (can be same as input for in-place operation)
 * @param size Vector size
 * @param params Optional parameters
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_apply_derivative_vector(ActivationType type, const double* input, 
                                      double* output, int size, 
                                      const ActivationParams* params);

/* =============================================================================
 * GENERIC ACTIVATION INTERFACE
 * ============================================================================= */

/**
 * @brief Generic activation function caller
 * @param type Activation function type
 * @param x Input value
 * @param params Optional parameters
 * @return Activated output value
 */
double activation_apply(ActivationType type, double x, const ActivationParams* params);

/**
 * @brief Generic activation derivative caller
 * @param type Activation function type
 * @param x Input value (or activated output for some functions)
 * @param params Optional parameters
 * @return Derivative value
 */
double activation_apply_derivative(ActivationType type, double x, 
                                  const ActivationParams* params);

/* =============================================================================
 * PARAMETER MANAGEMENT
 * ============================================================================= */

/**
 * @brief Initialize activation parameters with default values
 * @param params Parameters structure to initialize
 * @param type Activation function type
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_init_params(ActivationParams* params, ActivationType type);

/**
 * @brief Set parameter value by name
 * @param params Parameters structure
 * @param param_name Parameter name ("alpha", "beta", etc.)
 * @param value Parameter value
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_set_param(ActivationParams* params, const char* param_name, double value);

/**
 * @brief Get parameter value by name
 * @param params Parameters structure
 * @param param_name Parameter name
 * @param value Pointer to store parameter value
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_get_param(const ActivationParams* params, const char* param_name, 
                        double* value);

/**
 * @brief Validate parameters for given activation function
 * @param type Activation function type
 * @param params Parameters to validate
 * @return Error code (ACTIVATION_SUCCESS if valid)
 */
int activation_validate_params(ActivationType type, const ActivationParams* params);

/* =============================================================================
 * INFORMATION AND UTILITIES
 * ============================================================================= */

/**
 * @brief Get information about activation function
 * @param type Activation function type
 * @return Pointer to activation info structure (NULL if invalid type)
 */
const ActivationInfo* activation_get_info(ActivationType type);

/**
 * @brief Get activation function name
 * @param type Activation function type
 * @return Function name string (NULL if invalid type)
 */
const char* activation_get_name(ActivationType type);

/**
 * @brief Get activation function type from name
 * @param name Function name string
 * @return Activation function type (ACTIVATION_COUNT if not found)
 */
ActivationType activation_get_type_from_name(const char* name);

/**
 * @brief Check if activation function is bounded
 * @param type Activation function type
 * @return 1 if bounded, 0 if unbounded, -1 if invalid type
 */
int activation_is_bounded(ActivationType type);

/**
 * @brief Get output range of activation function
 * @param type Activation function type
 * @param min_val Pointer to store minimum value
 * @param max_val Pointer to store maximum value
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_get_range(ActivationType type, double* min_val, double* max_val);

/**
 * @brief Print activation function information
 * @param type Activation function type
 * @param file File stream to print to (NULL for stdout)
 */
void activation_print_info(ActivationType type, FILE* file);

/**
 * @brief Print all available activation functions
 * @param file File stream to print to (NULL for stdout)
 */
void activation_print_all(FILE* file);

/* =============================================================================
 * NUMERICAL STABILITY UTILITIES
 * ============================================================================= */

/**
 * @brief Clip value to prevent numerical overflow
 * @param x Input value
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @return Clipped value
 */
double activation_clip(double x, double min_val, double max_val);

/**
 * @brief Safe exponential function (prevents overflow)
 * @param x Input value
 * @return Safe exponential value
 */
double activation_safe_exp(double x);

/**
 * @brief Safe logarithm function (prevents underflow)
 * @param x Input value
 * @return Safe logarithm value
 */
double activation_safe_log(double x);

/**
 * @brief Check if value is in valid range for computations
 * @param x Input value
 * @return 1 if valid, 0 if invalid
 */
int activation_is_valid(double x);

/* =============================================================================
 * BENCHMARKING AND TESTING
 * ============================================================================= */

/**
 * @brief Benchmark activation function performance
 * @param type Activation function type
 * @param iterations Number of iterations
 * @param params Optional parameters
 * @return Time taken in seconds
 */
double activation_benchmark(ActivationType type, int iterations, 
                           const ActivationParams* params);

/**
 * @brief Test activation function with known values
 * @param type Activation function type
 * @param params Optional parameters
 * @param tolerance Tolerance for comparisons
 * @return Error code (ACTIVATION_SUCCESS if all tests pass)
 */
int activation_test(ActivationType type, const ActivationParams* params, 
                   double tolerance);

/**
 * @brief Run comprehensive tests for all activation functions
 * @param tolerance Tolerance for comparisons
 * @return Number of failed tests (0 if all pass)
 */
int activation_test_all(double tolerance);

/* =============================================================================
 * CUSTOM ACTIVATION FUNCTIONS
 * ============================================================================= */

/**
 * @brief Register custom activation function
 * @param func Custom activation function pointer
 * @param deriv_func Custom derivative function pointer
 * @param name Function name
 * @param params Custom parameters
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_register_custom(ActivationFunc func, ActivationDerivFunc deriv_func,
                              const char* name, void* params);

/**
 * @brief Apply custom activation function
 * @param x Input value
 * @param params Parameters containing custom function pointers
 * @return Custom activation output
 */
double activation_custom(double x, const ActivationParams* params);

/**
 * @brief Apply custom activation derivative
 * @param x Input value
 * @param params Parameters containing custom function pointers
 * @return Custom derivative output
 */
double activation_custom_derivative(double x, const ActivationParams* params);

/* =============================================================================
 * THREAD SAFETY
 * ============================================================================= */

/**
 * @brief Initialize thread-safe activation function library
 * @return Error code (ACTIVATION_SUCCESS on success)
 */
int activation_init_thread_safe(void);

/**
 * @brief Cleanup thread-safe activation function library
 */
void activation_cleanup_thread_safe(void);

/* =============================================================================
 * ERROR HANDLING
 * ============================================================================= */

/**
 * @brief Get error message for error code
 * @param error_code Error code
 * @return Error message string
 */
const char* activation_get_error_message(int error_code);

/**
 * @brief Set error handler callback
 * @param handler Error handler function pointer
 */
void activation_set_error_handler(void (*handler)(int error_code, const char* message));

#endif /* ACTIVATION_H */