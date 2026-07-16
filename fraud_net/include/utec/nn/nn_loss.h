#ifndef NN_LOSS_H
#define NN_LOSS_H
#include "tensor.h"
#include "nn_interfaces.h"

namespace utec::neural_network {
    template<typename T>
    class MSELoss final : public ILoss<T, 2> {
        private:
            utec::algebra::Tensor<T, 2> y_prediction_;
            utec::algebra::Tensor<T, 2> y_true_;
        public:
            MSELoss(const utec::algebra::Tensor<T, 2>& y_prediction, const utec::algebra::Tensor<T, 2>& y_true)
                : y_prediction_(y_prediction), y_true_(y_true) {}

            T loss() const override {
                T total_loss = 0;
                size_t total_elements = y_true_.shape()[0] * y_true_.shape()[1];
                for (size_t i = 0; i < y_true_.shape()[0]; ++i) {
                    for (size_t j = 0; j < y_true_.shape()[1]; ++j) {
                        total_loss += std::pow(y_prediction_(i, j) - y_true_(i, j), 2);
                    }
                }
                return total_loss / static_cast<T>(total_elements);
            }

            utec::algebra::Tensor<T, 2> loss_gradient() const override {
                utec::algebra::Tensor<T, 2> gradient(y_true_.shape());
                size_t total_elements = y_true_.shape()[0] * y_true_.shape()[1];
                for (size_t i = 0; i < y_true_.shape()[0]; ++i) {
                    for (size_t j = 0; j < y_true_.shape()[1]; ++j) {
                        gradient(i, j) = 2 * (y_prediction_(i, j) - y_true_(i, j)) / static_cast<T>(total_elements);
                    }
                }
                return gradient;
            }
    };
    template<typename T>
    class BCELoss final : public ILoss<T, 2> {
        private:
            utec::algebra::Tensor<T, 2> y_prediction_;
            utec::algebra::Tensor<T, 2> y_true_;
        public:
            BCELoss(const utec::algebra::Tensor<T, 2>& y_prediction, const utec::algebra::Tensor<T, 2>& y_true)
                : y_prediction_(y_prediction), y_true_(y_true) {}
            T loss() const override {
                T total_loss = 0;
                T epsilon = 1e-15; // Pequeño valor para evitar log(0)
                size_t total_elements = y_true_.shape()[0] * y_true_.shape()[1];
                for (size_t i = 0; i < y_true_.shape()[0]; ++i) {
                    for (size_t j = 0; j < y_true_.shape()[1]; ++j) {
                        total_loss += -y_true_(i, j) * std::log(y_prediction_(i, j) + epsilon) - (1 - y_true_(i, j)) * std::log(1 - y_prediction_(i, j) + epsilon);
                    }
                }
                return total_loss / static_cast<T>(total_elements);
            }

            utec::algebra::Tensor<T, 2> loss_gradient() const override {
                utec::algebra::Tensor<T, 2> gradient(y_true_.shape());
                T epsilon = 1e-15; // Pequeño valor para evitar divisiones por cero
                size_t total_elements = y_true_.shape()[0] * y_true_.shape()[1];
                for (size_t i = 0; i < y_true_.shape()[0]; ++i) {
                    for (size_t j = 0; j < y_true_.shape()[1]; ++j) {
                        gradient(i, j) = (y_prediction_(i, j) - y_true_(i, j)) / (y_prediction_(i, j) * (1 - y_prediction_(i, j)) + epsilon) / static_cast<T>(total_elements);
                    }
                }
                return gradient;
            }
    };
}
#endif // NN_LOSS_H