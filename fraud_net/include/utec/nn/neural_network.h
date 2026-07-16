#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "tensor.h"
#include "nn_interfaces.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <fstream>

namespace utec::neural_network {
    template<typename T>
    class NeuralNetwork {
        private:
            std::vector<std::unique_ptr<ILayer<T>>> layers_;
        public:
            void add_layer(std::unique_ptr<ILayer<T>> layer) {
                layers_.push_back(std::move(layer));
            }
            
            template <template <typename> class LossType, 
                    template <typename> class OptimizerType = SGD>
            void train(const utec::algebra::Tensor<T, 2>& X, const utec::algebra::Tensor<T, 2>& Y, 
                    const size_t epochs, const size_t batch_size, T learning_rate) {
                
                OptimizerType<T> optimizer(learning_rate);
                
                for (size_t epoch = 0; epoch < epochs; ++epoch) {
                    utec::algebra::Tensor<T, 2> input = X;
                    for (auto& layer : layers_) {
                        input = layer->forward(input);
                    }
                    
                    LossType<T> loss_function(input, Y);
                    
                    utec::algebra::Tensor<T, 2> grad = loss_function.loss_gradient();
                    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                        grad = (*it)->backward(grad);
                    }
                    
                    for (auto& layer : layers_) {
                        layer->update_params(optimizer);
                    }
                }
            }
            
            utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
                utec::algebra::Tensor<T, 2> input = X;
                for (auto& layer : layers_) {
                    input = layer->forward(input);
                }
                return input;
            }
            
            void save(const std::string& filename) const {
                std::ofstream os(filename);
                if (os.is_open()) {
                    for (const auto& layer : layers_) {
                        layer->save(os);
                    }
                }
            }

            void load(const std::string& filename) {
                std::ifstream is(filename);
                if (is.is_open()) {
                    for (auto& layer : layers_) {
                        layer->load(is);
                    }
                }
            }
    };
}
#endif // NEURAL_NETWORK_H