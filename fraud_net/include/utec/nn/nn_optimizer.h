#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H
#include "tensor.h"
#include "nn_interfaces.h"
#include <cmath>

using utec::algebra::Tensor;

namespace utec::neural_network {
    template<typename T>
    class SGD final : public IOptimizer<T> {
        private:
            T learning_rate_;
        public:
            explicit SGD(T learning_rate = 0.01) : learning_rate_(learning_rate) {}
            void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
                auto params_it = params.begin();
                auto grads_it = grads.begin();
                
                for (; params_it != params.end() && grads_it != grads.end(); ++params_it, ++grads_it) {
                    *params_it = *params_it - learning_rate_ * (*grads_it);
                }
            }
            
            void step() override {}
    };
    template<typename T>
    class Adam final : public IOptimizer<T> {
        private:
            T learning_rate_;
            T beta1_;
            T beta2_;
            T epsilon_;
            Tensor<T, 2> m_;
            Tensor<T, 2> v_;
            size_t t_;
            bool initialized_;
            
        public:
            explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8) 
                : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), 
                  t_(0), initialized_(false) {}
            
            void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
                if (!initialized_) {
                    m_ = Tensor<T, 2>(params.shape());
                    v_ = Tensor<T, 2>(params.shape());
                    m_.fill(T(0));
                    v_.fill(T(0));
                    initialized_ = true;
                }
                
                t_++;
                
                auto params_it = params.begin();
                auto grads_it = grads.begin();
                auto m_it = m_.begin();
                auto v_it = v_.begin();
                
                for (; params_it != params.end(); ++params_it, ++grads_it, ++m_it, ++v_it) {
                    *m_it = beta1_ * (*m_it) + (1 - beta1_) * (*grads_it);
                    
                    *v_it = beta2_ * (*v_it) + (1 - beta2_) * (*grads_it) * (*grads_it);
                    
                    T m_hat = (*m_it) / (1 - std::pow(beta1_, static_cast<T>(t_)));
                    
                    T v_hat = (*v_it) / (1 - std::pow(beta2_, static_cast<T>(t_)));
                    
                    *params_it = *params_it - learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }
            
            void step() override {}
    };
}

#endif // NN_OPTIMIZER_H