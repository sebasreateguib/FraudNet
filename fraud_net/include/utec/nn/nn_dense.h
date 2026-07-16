#ifndef NN_DENSE_H
#define NN_DENSE_H
#include "nn_interfaces.h"
#include "tensor.h"

using utec::algebra::Tensor;

namespace utec::neural_network {
    template<typename T>
    class Dense final : public ILayer<T> {
    private:
        Tensor<T, 2> weights_;
        Tensor<T, 2> bias_;
        Tensor<T, 2> input_cache_;
        Tensor<T, 2> dW_;
        Tensor<T, 2> db_;
        size_t in_features_;
        size_t out_features_;
    public:
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) 
            : in_features_(in_f), out_features_(out_f) {
            
            weights_ = Tensor<T, 2>(in_features_, out_features_);
            init_w_fun(weights_);
            
            bias_ = Tensor<T, 2>(1, out_features_);
            init_b_fun(bias_);
            
            dW_ = Tensor<T, 2>(in_features_, out_features_);
            db_ = Tensor<T, 2>(1, out_features_);
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            input_cache_ = x;
            auto output = utec::algebra::matrix_product(x, weights_);
            output = output + bias_;
            return output;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
            auto weights_T = utec::algebra::transpose_2d(weights_);
            auto dX = utec::algebra::matrix_product(dZ, weights_T);
            
            auto input_T = utec::algebra::transpose_2d(input_cache_);
            dW_ = utec::algebra::matrix_product(input_T, dZ);
            
            db_.fill(T{0});
            const auto& dZ_shape = dZ.shape();
            size_t batch_size = dZ_shape[0];
            size_t out_features = dZ_shape[1];
            
            for (size_t j = 0; j < out_features; ++j) {
                T sum = T{0};
                for (size_t i = 0; i < batch_size; ++i) {
                    sum += dZ(i, j);
                }
                db_(0, j) = sum;
            }
            
            return dX;
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(weights_, dW_);
            optimizer.update(bias_, db_);
        }
        
        void save(std::ostream& os) const override {
            for (size_t i = 0; i < weights_.shape()[0]; ++i)
                for (size_t j = 0; j < weights_.shape()[1]; ++j)
                    os << weights_(i, j) << " ";
            for (size_t i = 0; i < bias_.shape()[0]; ++i)
                for (size_t j = 0; j < bias_.shape()[1]; ++j)
                    os << bias_(i, j) << " ";
        }

        void load(std::istream& is) override {
            for (size_t i = 0; i < weights_.shape()[0]; ++i)
                for (size_t j = 0; j < weights_.shape()[1]; ++j)
                    is >> weights_(i, j);
            for (size_t i = 0; i < bias_.shape()[0]; ++i)
                for (size_t j = 0; j < bias_.shape()[1]; ++j)
                    is >> bias_(i, j);
        }
    };
}
#endif // NN_DENSE_H