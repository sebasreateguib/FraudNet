#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H
#include "tensor.h"
#include "nn_interfaces.h"

namespace utec::neural_network {
    template<typename T>
    class ReLU final : public ILayer<T> {
    private:
        utec::algebra::Tensor<T, 2> z_;
    public:
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
            z_ = z;
            utec::algebra::Tensor<T, 2> result(z.shape());
            for (size_t i = 0; i < z.shape()[0]; ++i) {
                for (size_t j = 0; j < z.shape()[1]; ++j) {
                    result(i, j) = std::max(z(i, j), T(0));
                }
            }
            return result;
        }
        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& g) override {
            utec::algebra::Tensor<T, 2> result(g.shape());
            for (size_t i = 0; i < g.shape()[0]; ++i) {
                for (size_t j = 0; j < g.shape()[1]; ++j) {
                    result(i, j) = (z_(i, j) > T(0)) ? g(i, j) : T(0);
                }
            }
            return result;
        }
        void update_params(IOptimizer<T>& optimizer) override {}
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
    private:
        utec::algebra::Tensor<T, 2> z_;
        utec::algebra::Tensor<T, 2> a_;
    public:
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
            z_ = z;
            utec::algebra::Tensor<T, 2> result(z.shape());
            for (size_t i = 0; i < z.shape()[0]; ++i) {
                for (size_t j = 0; j < z.shape()[1]; ++j) {
                    result(i, j) = T(1) / (T(1) + std::exp(-z(i, j)));
                }
            }
            a_ = result;
            return result;
        }
        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& g) override {
            utec::algebra::Tensor<T, 2> result(g.shape());
            for (size_t i = 0; i < g.shape()[0]; ++i) {
                for (size_t j = 0; j < g.shape()[1]; ++j) {
                    result(i, j) = g(i, j) * a_(i, j) * (T(1) - a_(i, j));
                }
            }
            return result;
        }
        void update_params(IOptimizer<T>& optimizer) override {
        }
    };
}
#endif // NN_ACTIVATION_H