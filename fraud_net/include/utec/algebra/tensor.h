#ifndef TENSOR_H
#define TENSOR_H
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <numeric>

namespace utec::algebra {
template <size_t Rank>
size_t flatten_index(const std::array<size_t, Rank>& shape, const std::array<size_t, Rank>& idxs);
    template <typename T, size_t Rank>
    class Tensor {
        private:
            std::array<size_t, Rank> shape_;
            std::vector<T> data_;
            static size_t total_size_from_array(const std::array<size_t, Rank>& shape) {
                size_t n = 1;
                for (auto d : shape) n *= d;
                return n;
            }
            template<typename... Dims>
            void set_shape(std::array<size_t, Rank>& arr, Dims... dims) {
                size_t tmp[] = { static_cast<size_t>(dims)... };
                for (size_t i = 0; i < Rank; ++i) arr[i] = tmp[i];
            }
            void imprimir(std::ostream& os, size_t dim, size_t& idx) const {
                if (dim == Rank - 1) {
                    for (size_t i = 0; i < shape_[dim]; ++i) {
                        os << data_[idx++];
                        if (i + 1 < shape_[dim]) os << " ";
                    }
                } else {
                    os << "{\n";
                    for (size_t i = 0; i < shape_[dim]; ++i) {
                        imprimir(os, dim + 1, idx);
                        os << "\n";
                    }
                    os << "}";
                }
            }
            static std::array<size_t, Rank> broadcast_shape(const std::array<size_t, Rank>& a, const std::array<size_t, Rank>& b) {
                std::array<size_t, Rank> result;
                for (size_t i = 0; i < Rank; ++i) {
                    if (a[i] == b[i]) result[i] = a[i];
                    else if (a[i] == 1) result[i] = b[i];
                    else if (b[i] == 1) result[i] = a[i];
                    else throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
                }
                return result;
            }
            static size_t broadcast_index(const std::array<size_t, Rank>& shape, const std::array<size_t, Rank>& idxs) {
                size_t offset = 0, stride = 1;
                for (ptrdiff_t i = Rank - 1; i >= 0; --i) {
                    size_t idx = (shape[i] == 1) ? 0 : idxs[i];
                    offset += idx * stride;
                    stride *= shape[i];
                }
                return offset;
            }
        public:
            Tensor() = default;
            Tensor(const std::array<size_t, Rank>& shape) : shape_(shape), data_(total_size_from_array(shape)) {}
            template <typename... Dims>
            Tensor(Dims... dims) {
                if (sizeof...(Dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
                }
                set_shape(shape_, dims...);
                data_.resize(total_size_from_array(shape_));
            }

            // Iteradores
            auto begin() { return data_.begin(); }
            auto end() { return data_.end(); }
            auto begin() const { return data_.cbegin(); }
            auto end() const { return data_.cend(); }
            auto cbegin() const { return data_.cbegin(); }
            auto cend() const { return data_.cend(); }

            template <typename... Idxs>
            T& operator()(Idxs... idxs) {
                static_assert(sizeof...(Idxs) == Rank, "Invalid number of indices");
                std::array<size_t, Rank> idx_array{static_cast<size_t>(idxs)...};
                size_t offset = flatten_index(shape_, idx_array);
                return data_[offset];
            }
            template <typename... Idxs>
            const T& operator()(Idxs... idxs) const {
                static_assert(sizeof...(Idxs) == Rank, "Número de índices incorrecto");
                std::array<size_t, Rank> idx_array{static_cast<size_t>(idxs)...};
                size_t offset = flatten_index(shape_, idx_array);
                return data_[offset];
            }

            const std::array<size_t, Rank>& shape() const noexcept { return shape_; }
            size_t size() const noexcept { return data_.size(); }  // AGREGADO: método size()
            void fill(const T& value) noexcept { std::fill(data_.begin(), data_.end(), value); }

            Tensor& operator=(std::initializer_list<T> values) {
                if (values.size() != data_.size()) {
                    throw std::invalid_argument("Data size does not match tensor size");
                }
                std::copy(values.begin(), values.end(), data_.begin());
                return *this;
            }

            friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
                size_t idx = 0;
                t.imprimir(os, 0, idx);
                return os;
            }

            template <typename... Dims>
            void reshape(Dims... dims) {
                if (sizeof...(Dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
                }
                std::array<size_t, Rank> new_shape;
                size_t tmp[] = { static_cast<size_t>(dims)... };
                size_t total = 1;
                for (size_t i = 0; i < Rank; ++i) {
                    new_shape[i] = tmp[i];
                    total *= tmp[i];
                }
                if (total != data_.size()) {
                    data_.resize(total, T{});
                }
                shape_ = new_shape;
            }

            Tensor operator+(const Tensor& other) const {
                auto out_shape = broadcast_shape(shape_, other.shape_);
                Tensor result(out_shape);

                std::array<size_t, Rank> idxs{};
                for (size_t flat = 0; flat < result.data_.size(); ++flat) {
                    size_t rem = flat;
                    for (ptrdiff_t i = Rank - 1; i >= 0; --i) {
                        idxs[i] = rem % out_shape[i];
                        rem /= out_shape[i];
                    }
                    size_t idx_a = broadcast_index(shape_, idxs);
                    size_t idx_b = broadcast_index(other.shape_, idxs);
                    result.data_[flat] = data_[idx_a] + other.data_[idx_b];
                }
                return result;
            }

            Tensor operator-(const Tensor& other) const {
                auto out_shape = broadcast_shape(shape_, other.shape_);
                Tensor result(out_shape);

                std::array<size_t, Rank> idxs{};
                for (size_t flat = 0; flat < result.data_.size(); ++flat) {
                    size_t rem = flat;
                    for (ptrdiff_t i = Rank - 1; i >= 0; --i) {
                        idxs[i] = rem % out_shape[i];
                        rem /= out_shape[i];
                    }
                    size_t idx_a = broadcast_index(shape_, idxs);
                    size_t idx_b = broadcast_index(other.shape_, idxs);
                    result.data_[flat] = data_[idx_a] - other.data_[idx_b];
                }
                return result;
            }

            Tensor operator*(const Tensor& other) const {
                auto out_shape = broadcast_shape(shape_, other.shape_);
                Tensor result(out_shape);

                std::array<size_t, Rank> idxs{};
                for (size_t flat = 0; flat < result.data_.size(); ++flat) {
                    size_t rem = flat;
                    for (ptrdiff_t i = Rank - 1; i >= 0; --i) {
                        idxs[i] = rem % out_shape[i];
                        rem /= out_shape[i];
                    }
                    size_t idx_a = broadcast_index(shape_, idxs);
                    size_t idx_b = broadcast_index(other.shape_, idxs);
                    result.data_[flat] = data_[idx_a] * other.data_[idx_b];
                }
                return result;
            }

            Tensor operator/(const Tensor& other) const {
                auto out_shape = broadcast_shape(shape_, other.shape_);
                Tensor result(out_shape);

                std::array<size_t, Rank> idxs{};
                for (size_t flat = 0; flat < result.data_.size(); ++flat) {
                    size_t rem = flat;
                    for (ptrdiff_t i = Rank - 1; i >= 0; --i) {
                        idxs[i] = rem % out_shape[i];
                        rem /= out_shape[i];
                    }
                    size_t idx_a = broadcast_index(shape_, idxs);
                    size_t idx_b = broadcast_index(other.shape_, idxs);
                    result.data_[flat] = data_[idx_a] / other.data_[idx_b];
                }
                return result;
            }

            Tensor operator+(const T& scalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data_.size(); ++i)
                    result.data_[i] = data_[i] + scalar;
                return result;
            }
            Tensor operator-(const T& scalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data_.size(); ++i)
                    result.data_[i] = data_[i] - scalar;
                return result;
            }
            Tensor operator*(const T& scalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data_.size(); ++i)
                    result.data_[i] = data_[i] * scalar;
                return result;
            }
            Tensor operator/(const T& scalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data_.size(); ++i)
                    result.data_[i] = data_[i] / scalar;
                return result;
            }
    };

    template <typename T, size_t Rank>
    Tensor<T, Rank - 1> sum(const Tensor<T, Rank>& tensor, size_t axis) {
        std::array<size_t, Rank - 1> new_shape;
        size_t idx = 0;
        for (size_t i = 0; i < Rank; ++i) {
            if (i != axis) {
                new_shape[idx++] = tensor.shape()[i];
            }
        }
        Tensor<T, Rank - 1> result(new_shape);
        for (size_t i = 0; i < result.size(); ++i) {
            result.data_[i] = std::accumulate(tensor.data_.begin() + i * tensor.shape()[axis],
                                              tensor.data_.begin() + (i + 1) * tensor.shape()[axis], T(0));
        }
        return result;
    }
    template <size_t Rank>
    size_t flatten_index(const std::array<size_t, Rank>& shape, const std::array<size_t, Rank>& idxs) {
        size_t offset = 0, stride = 1;
        for (ptrdiff_t i = Rank - 1; i >= 0; --i) {
            offset += idxs[i] * stride;
            stride *= shape[i];
        }
        return offset;
    }

    template <typename T, size_t Rank>
    Tensor<T, Rank> operator+(const T& scalar, const Tensor<T, Rank>& tensor) {
        return tensor + scalar;
    }

    template<typename T, size_t Rank>
    Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& t) {
        if constexpr (Rank < 2) {
            throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        } else {
            std::array<size_t, Rank> new_shape = t.shape();
            std::swap(new_shape[Rank - 2], new_shape[Rank - 1]);
            Tensor<T, Rank> result(new_shape);

            std::array<size_t, Rank> idx{};
            for (size_t flat = 0; flat < result.cend() - result.cbegin(); ++flat) {
                size_t rem = flat;
                for (ptrdiff_t i = Rank - 1; i >= 0; --i) {
                    idx[i] = rem % new_shape[i];
                    rem /= new_shape[i];
                }
                auto orig_idx = idx;
                std::swap(orig_idx[Rank - 2], orig_idx[Rank - 1]);
                result.begin()[flatten_index<Rank>(new_shape, idx)] =
                    t.begin()[flatten_index<Rank>(t.shape(), orig_idx)];
            }
            return result;
        }
    }
    template <typename T, size_t Rank>
    Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
        static_assert(Rank == 2 || Rank == 3, "Only supports Rank 2 or 3 tensors");
        
        if constexpr (Rank == 2) {
            size_t m = a.shape()[0];
            size_t k = a.shape()[1];
            size_t n = b.shape()[1];
            
            if (a.shape()[1] != b.shape()[0]) {
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
            }
            
            Tensor<T, 2> result(m, n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    T sum = T{};
                    for (size_t p = 0; p < k; ++p) {
                        sum += a(i, p) * b(p, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        } else {
            size_t batch = a.shape()[0];
            size_t m = a.shape()[1];
            size_t k = a.shape()[2];
            size_t n = b.shape()[2];
            
            if (a.shape()[2] != b.shape()[1]) {
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
            }
            if (a.shape()[0] != b.shape()[0]) {
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
            
            Tensor<T, 3> result(batch, m, n);
            for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        T sum = T{};
                        for (size_t p = 0; p < k; ++p) {
                            sum += a(b_idx, i, p) * b(b_idx, p, j);
                        }
                        result(b_idx, i, j) = sum;
                    }
                }
            }
            return result;
        }
    }
}
#endif // TENSOR_H