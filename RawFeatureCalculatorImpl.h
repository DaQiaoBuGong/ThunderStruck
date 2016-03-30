/**
 * thunderstruck/tracker: RawFeatureCalculatorImpl.h
 */

#ifndef H_THUNDERSTRUCK_RAWFEATURECALCULATORIMPL
#define H_THUNDERSTRUCK_RAWFEATURECALCULATORIMPL

#include <cuda_runtime.h>

namespace thunderstruck {

//#################### CONSTANTS ####################

/** The square root of the number of raw features to calculate per sample. */
const size_t PATCH_SIZE = 16;

/** The number of raw features to calculate per sample. */
const size_t RAW_FEATURE_COUNT = PATCH_SIZE * PATCH_SIZE;

//#################### WRAPPER FUNCTION DECLARATIONS ####################

/**
 * \brief Calculates raw features for each sample that is completely within the bounds of the specified frame.
 *
 * \param[in]  frame            The texture object representing the frame on the GPU.
 * \param[in]  sampleData       A linearised version of the sample points (expressed relative to the current bounding box).
 * \param[in]  keepSamples      An array of flags specifying whether or not each sample is within the bounds of the frame.
 * \param[in]  sampleCount      The total number of samples (including those that fall outside the bounds of the frame).
 * \param[in]  bbX              The x coordinate of the current bounding box's anchor point.
 * \param[in]  bbY              The y coordinate of the current bounding box's anchor point.
 * \param[in]  bbWidth          The width of the current bounding box.
 * \param[in]  bbHeight         The height of the current bounding box.
 * \param[out] features         The array into which the calculated features should be written.
 * \param[in]  offset           The offset from the start of the features array at which to start writing.
 * \param[in]  stride           The stride in the features array between one sample and the next.
 * \param[in]  threadsPerBlock  The number of threads to use in each CUDA block.
 */
extern "C" void calculate_raw_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                       float bbX, float bbY, float bbWidth, float bbHeight,
                                       double *features, size_t offset, size_t stride,
                                       int threadsPerBlock);

}

#endif
