/**
 * thunderstruck/tracker: HaarFeatureCalculatorImpl.h
 */

#ifndef H_THUNDERSTRUCK_HAARFEATURECALCULATORIMPL
#define H_THUNDERSTRUCK_HAARFEATURECALCULATORIMPL

#include <cuda_runtime.h>

namespace thunderstruck {

//#################### CONSTANTS ####################

/** The number of Haar features that will be calculated per sample (16 boxes at each of 2 different scales, 6 types of feature). */
const size_t HAAR_FEATURE_COUNT = 192;

//#################### WRAPPER FUNCTION DECLARATIONS ####################

/**
 * \brief Calculates Haar features for each sample that is completely within the bounds of the specified frame.
 *
 * \param[in]  integralFrame    The texture object representing the integral image for the frame on the GPU.
 * \param[in]  sampleData       A linearised version of the sample points (expressed relative to the current bounding box).
 * \param[in]  keepSamples      An array of flags specifying whether or not each sample is within the bounds of the frame.
 * \param[in]  sampleCount      The total number of samples (including those that fall outside the bounds of the frame).
 * \param[in]  bbX              The x coordinate of the current bounding box's anchor point.
 * \param[in]  bbY              The y coordinate of the current bounding box's anchor point.
 * \param[in]  bbWidth          The width of the current bounding box.
 * \param[in]  bbHeight         The height of the current bounding box.
 * \param[in]  bottoms          The bottom coordinates of the mini-boxes for the Haar features.
 * \param[in]  lefts            The left coordinates of the mini-boxes for the Haar features.
 * \param[in]  rights           The right coordinates of the mini-boxes for the Haar features.
 * \param[in]  tops             The top coordinates of the mini-boxes for the Haar features.
 * \param[in]  weights          The weights of the mini-boxes for the Haar features.
 * \param[out] features         The array into which the calculated features should be written.
 * \param[in]  offset           The offset from the start of the features array at which to start writing.
 * \param[in]  stride           The stride in the features array between one sample and the next.
 */
extern "C" void calculate_haar_features(cudaTextureObject_t integralFrame,
                                        float *sampleData, int *keepSamples, size_t sampleCount,
                                        float bbX, float bbY, float bbWidth, float bbHeight,
                                        float *bottoms, float *lefts, float *rights, float *tops, float *weights,
                                        double *features, size_t offset, size_t stride);

}

#endif
