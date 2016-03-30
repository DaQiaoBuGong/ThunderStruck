/**
 * thunderstruck/util: GPUVector.h
 */

#ifndef H_THUNDERSTRUCK_GPUVECTOR
#define H_THUNDERSTRUCK_GPUVECTOR

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include <boost/serialization/shared_ptr.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace thunderstruck {

/**
 * \brief An instance of (an instantiation of) this template can be used to represent a vector of data on the GPU.
 */
template <typename T>
class GPUVector
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** A pointer to the vector data on the GPU. */
  boost::shared_ptr<T> m_gpuPtr;

  /** The size of the vector. */
  size_t m_size;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Default constructor (for convenience only).
   */
  GPUVector()
  : m_size(0)
  {}

  /**
   * \brief Constructs an empty GPU vector of the specified size.
   *
   * \param[in] size  The size of the vector.
   */
  explicit GPUVector(size_t size)
  : m_size(size)
  {
    T *gpuPtr = NULL;
    checkCudaErrors(cudaMalloc((void**)&gpuPtr, m_size * sizeof(T)));
    m_gpuPtr.reset(gpuPtr, cudaFree);
  }

  /**
   * \brief Constructs a GPU vector that contains n copies of the specified value (this involves a data transfer).
   *
   * \param[in] n     The size of the vector.
   * \param[in] value The value to which to set the vector's elements.
   */
  GPUVector(size_t n, const T& value)
  {
    initialise_from(std::vector<T>(n, value));
  }

  /**
   * \brief Constructs a GPU vector from a CPU vector (this involves a data transfer).
   *
   * \param[in] cpuVec  The CPU vector.
   */
  explicit GPUVector(const std::vector<T>& cpuVec)
  {
    initialise_from(cpuVec);
  }

  /**
   * \brief Constructs a GPU vector from a pointer to some existing memory on the GPU and a size.
   *
   * Note that the resulting GPU vector explicitly does not take responsibility for deallocating
   * the memory later (that is, ownership is not transferred to the GPU vector).
   *
   * \param[in] gpuPtr  The pointer to the memory on the GPU.
   * \param[in] size    The size of the GPU vector.
   */
  GPUVector(T *gpuPtr, size_t size)
  : m_size(size)
  {
    m_gpuPtr.reset(gpuPtr, boost::serialization::null_deleter());
  }

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Sets all elements of the vector to the specified value.
   *
   * \param[in] value The value to which to set the elements.
   */
  void fill(const T& value)
  {
    std::vector<T> cpuVec(m_size, value);
    checkCudaErrors(cudaMemcpy(m_gpuPtr.get(), &cpuVec[0], m_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  /**
   * \brief Gets a pointer to the vector data on the GPU.
   *
   * \return  A pointer to the vector data on the GPU.
   */
  T *get() const
  {
    return m_gpuPtr.get();
  }

  /**
   * \brief Gets the value of the specified element of the vector.
   *
   * \param[in] index An index indicating the element of the vector to get.
   * \return          The value of the element at the specified index.
   */
  T get(size_t index) const
  {
    T result;
    checkCudaErrors(cudaMemcpy(&result, m_gpuPtr.get() + index, sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }

  /**
   * \brief Outputs the vector data to a stream (for debugging purposes).
   *
   * \param[in] os    The stream to which to output the data.
   * \param[in] limit The maximum number of elements to output (starting from the beginning of the vector).
   */
  void output(std::ostream& os, size_t limit = INT_MAX) const
  {
    std::vector<T> cpuVec = to_cpu();
#if 0
    std::cout << std::fixed << std::setprecision(17);
#endif
    std::copy(cpuVec.begin(), limit != INT_MAX ? cpuVec.begin() + std::min(limit, cpuVec.size()) : cpuVec.end(), std::ostream_iterator<T>(os, " "));
    os << '\n';
  }

  /**
   * \brief Sets the specified element of the vector to the specified value.
   *
   * \param[in] index An index indicating the element of the vector to set.
   * \param[in] value The value to which to set the element.
   */
  void set(size_t index, const T& value)
  {
    checkCudaErrors(cudaMemcpy(m_gpuPtr.get() + index, &value, sizeof(T), cudaMemcpyHostToDevice));
  }

  /**
   * \brief Gets the size of the vector.
   *
   * \return  The size of the vector.
   */
  size_t size() const
  {
    return m_size;
  }

  /**
   * \brief Constructs a CPU vector corresponding to this vector (this involves a data transfer).
   *
   * \return  A CPU vector corresponding to this vector.
   */
  std::vector<T> to_cpu() const
  {
    std::vector<T> cpuVec(m_size);
    checkCudaErrors(cudaMemcpy(&cpuVec[0], m_gpuPtr.get(), m_size * sizeof(T), cudaMemcpyDeviceToHost));
    return cpuVec;
  }

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Initialises this GPU vector from a CPU vector.
   *
   * \param[in] cpuVec  The CPU vector.
   */
  void initialise_from(const std::vector<T>& cpuVec)
  {
    m_size = cpuVec.size();
    T *gpuPtr = NULL;
    checkCudaErrors(cudaMalloc((void**)&gpuPtr, m_size * sizeof(T)));
    checkCudaErrors(cudaMemcpy(gpuPtr, &cpuVec[0], m_size * sizeof(T), cudaMemcpyHostToDevice));
    m_gpuPtr.reset(gpuPtr, cudaFree);
  }
};

//#################### HELPER FUNCTIONS ####################

/**
 * \brief Constructs a GPU vector from a CPU vector (this involves a data transfer).
 *
 * Use of this function template is preferable to using the GPUVector constructor
 * because it allows T to be automatically deduced.
 *
 * \param[in] cpuVec  The CPU vector.
 * \return            The GPU vector.
 */
template <typename T>
GPUVector<T> make_gpu_vector(const std::vector<T>& cpuVec)
{
  return GPUVector<T>(cpuVec);
}

}

#endif
