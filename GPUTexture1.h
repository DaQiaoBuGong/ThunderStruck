/**
 * thunderstruck/util: GPUTexture1.h
 */

#ifndef H_THUNDERSTRUCK_GPUTEXTURE1
#define H_THUNDERSTRUCK_GPUTEXTURE1

#include <boost/shared_ptr.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <opencv2/opencv.hpp>

namespace thunderstruck {

/**
 * \brief An instance of (an instantiation of) this class template can be used to represent a single-channel texture on the GPU.
 */
template <typename T>
class GPUTexture1
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The height of the texture. */
  int m_height;

  /** A pointer to the texture object associated with this texture. */
  boost::shared_ptr<cudaTextureObject_t> m_tex;

  /** The width of the texture. */
  int m_width;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a GPU texture from an OpenCV image on the CPU (this involves a data transfer).
   *
   * \param[in] image The OpenCV image on the CPU.
   */
  explicit GPUTexture1(const cv::Mat_<T>& image)
  : m_height(image.rows), m_width(image.cols)
  {
    // Transfer the image data across to a CUDA array on the GPU.
    const int elemSize = image.elemSize1();
    const int bitDepth = elemSize * 8;
    cudaArray *buffer;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bitDepth, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&buffer, &channelDesc, image.cols, image.rows));
    checkCudaErrors(cudaMemcpy2DToArray(buffer, 0, 0, image.ptr(), image.cols * elemSize, image.cols * elemSize, image.rows, cudaMemcpyHostToDevice));

    // Create the texture object.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = buffer;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    m_tex.reset(new cudaTextureObject_t, free_texture_object);
    checkCudaErrors(cudaCreateTextureObject(m_tex.get(), &resDesc, &texDesc, NULL));
  }

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the texture object associated with this texture.
   *
   * \return  The texture object associated with this texture.
   */
  const cudaTextureObject_t& get() const
  {
    assert(m_tex);
    return *m_tex;
  }

  /**
   * \brief Gets the height of the texture.
   *
   * \return  The height of the texture.
   */
  int height() const
  {
    return m_height;
  }

  /**
   * \brief Constructs an OpenCV image corresponding to this texture (this involves a data transfer).
   *
   * \return  An OpenCV image corresponding to this texture.
   */
  cv::Mat_<T> to_cpu() const
  {
    cv::Mat_<T> image = cv::Mat_<T>::zeros(m_height, m_width);
    cudaResourceDesc resDesc;
    checkCudaErrors(cudaGetTextureObjectResourceDesc(&resDesc, *m_tex));
    const int elemSize = image.elemSize1();
    checkCudaErrors(cudaMemcpy2DFromArray(image.ptr(), m_width * elemSize, resDesc.res.array.array, 0, 0, m_width * elemSize, m_height, cudaMemcpyDeviceToHost));
    return image;
  }

  /**
   * \brief Gets the width of the texture.
   *
   * \return  The width of the texture.
   */
  int width() const
  {
    return m_width;
  }

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Frees the specified CUDA texture object and its associated array.
   *
   * \param[in] p A pointer to the CUDA texture object to free.
   */
  static void free_texture_object(cudaTextureObject_t *p)
  {
    if(p == NULL) return;
    const cudaTextureObject_t& tex = *p;
    cudaResourceDesc resDesc;
    checkCudaErrors(cudaGetTextureObjectResourceDesc(&resDesc, tex));
    checkCudaErrors(cudaFreeArray(resDesc.res.array.array));
    checkCudaErrors(cudaDestroyTextureObject(tex));
  }
};

//#################### TYPEDEFS ####################

typedef GPUTexture1<unsigned char> GPUTexture1b;
typedef GPUTexture1<int> GPUTexture1i;

}

#endif
