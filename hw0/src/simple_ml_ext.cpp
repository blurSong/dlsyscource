#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath> 
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float* X, const unsigned char* y,
    float* theta, size_t m, size_t n, size_t k,
    float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

     /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
        if (i + batch > m) {
            batch = m - i;
        }
        float* logits = new float[batch * k];
        float* gradients = new float[n * k];
        memset(logits, 0, batch * k * sizeof(float));
        memset(gradients, 0, n * k * sizeof(float));

        // Compute logits=softmax(X*theta)
        for (size_t bb = 0;bb < batch;bb++) {
            float exp_sum = 0;
            float log_max = -1e10;
            for (size_t kk = 0;kk < k;kk++) {
                for (size_t nn = 0;nn < n;nn++) {
                    logits[bb * k + kk] += X[(i + bb) * n + nn] * theta[nn * k + kk];
                }
                log_max = std::max(log_max, logits[bb * k + kk]);
            }
            for (size_t kk = 0;kk < k;kk++) {
                logits[bb * k + kk] = exp(logits[bb * k + kk] - log_max);
                exp_sum += logits[bb * k + kk];
            }
            for (size_t kk = 0;kk < k;kk++) {
                logits[bb * k + kk] /= exp_sum;
            }
        }
        // Compute gradients = X.T * (logits - y) nb.bk
        for (size_t nn = 0;nn < n;nn++) {
            for (size_t kk = 0;kk < k;kk++) {
                for (size_t bb = 0;bb < batch;bb++) {
                    gradients[nn * k + kk] += X[(i + bb) * n + nn] * (logits[bb * k + kk] - (y[i + bb] == kk));
                }
                gradients[nn * k + kk] /= batch;
            }
        }
        // Update theta
        for (size_t nn = 0;nn < n;nn++) {
            for (size_t kk = 0;kk < k;kk++) {
                theta[nn * k + kk] -= lr * gradients[nn * k + kk];
            }
        }
        delete[] logits;
        delete[] gradients;

        /// END YOUR CODE
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
            py::array_t<unsigned char, py::array::c_style> y,
            py::array_t<float, py::array::c_style> theta,
            float lr,
            int batch) {
                softmax_regression_epoch_cpp(
                    static_cast<const float*>(X.request().ptr),
                    static_cast<const unsigned char*>(y.request().ptr),
                    static_cast<float*>(theta.request().ptr),
                    X.request().shape[0],
                    X.request().shape[1],
                    theta.request().shape[1],
                    lr,
                    batch
                );
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
            py::arg("lr"), py::arg("batch"));
}
