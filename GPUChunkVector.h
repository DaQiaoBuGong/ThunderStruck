/**
 * thunderstruck/util: GPUChunkVector.h
 */

#ifndef H_THUNDERSTRUCK_GPUCHUNKVECTOR
#define H_THUNDERSTRUCK_GPUCHUNKVECTOR

#include "GPUVector.h"

namespace thunderstruck {

/**
 * \brief An instance of (an instantiation of) this template can be used to represent a GPU vector
          that is divided into a number of fixed-size chunks.
 */
template <typename T>
class GPUChunkVector
{
  //#################### PRIVATE VARIBLES ####################
private:
  /** The GPU buffer backing the chunk vector. */
  GPUVector<T> m_buffer;

  /** The number of chunks in the vector. */
  size_t m_chunkCount;

  /** The number of elements in each chunk. */
  size_t m_chunkSize;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Default constructor (for convenience only).
   */
  GPUChunkVector()
  : m_chunkCount(0), m_chunkSize(0)
  {}

  /**
   * \brief Constructs a GPU chunk vector.
   *
   * \param[in] chunkCount  The number of chunks in the vector.
   * \param[in] chunkSize   The number of elements in each chunk.
   */
  GPUChunkVector(size_t chunkCount, size_t chunkSize)
  : m_buffer(chunkCount * chunkSize), m_chunkCount(chunkCount), m_chunkSize(chunkSize)
  {}

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets a pointer to the entire vector.
   *
   * \return  A pointer to the entire vector.
   */
  T *get() const
  {
    return m_buffer.get();
  }

  /**
   * \brief Gets the specified chunk.
   *
   * \param[in] n The index of the chunk to get.
   * \return      The specified chunk.
   */
  GPUVector<T> get_chunk(size_t n) const
  {
    assert(n < m_chunkCount);
    return GPUVector<T>(m_buffer.get() + n * m_chunkSize, m_chunkSize);
  }
};

}

#endif
