#ifndef NN_INTERFACES_H
#define NN_INTERFACES_H
#include "tensor.h"
#include <iostream>

namespace utec::neural_network {
template <typename T>
    class IOptimizer {
        public:
            virtual ~IOptimizer() = default;
            virtual void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) = 0;
            virtual void step() = 0;
    };

template <typename T>
    class ILayer {
        public:
            virtual ~ILayer() = default;
            virtual utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) = 0;
            virtual utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) = 0;
            virtual void update_params(IOptimizer<T>& optimizer) = 0;
            
            // Serialización del modelo (Epic 5)
            virtual void save(std::ostream& os) const {}
            virtual void load(std::istream& is) {}
    };
template <typename T, size_t Rank>
    class ILoss {
        public:
            virtual ~ILoss() = default;
            virtual T loss() const = 0;
            virtual utec::algebra::Tensor<T, Rank> loss_gradient() const = 0;
    };
}
#endif // NN_INTERFACES_H