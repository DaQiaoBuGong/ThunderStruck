/**
 * thunderstruck/tracker: SampleFilterer.h
 */

#ifndef H_THUNDERSTRUCK_SAMPLEFILTERER
#define H_THUNDERSTRUCK_SAMPLEFILTERER

namespace thunderstruck {

//#################### WRAPPER FUNCTION DECLARATIONS ####################

/**
 * \brief Filters a set of sample bounding boxes (represented by their top-left anchor points, with x and y
 *        coordinates expressed relative to the current bounding box), keeping only those that fall entirely
 *        within the frame window.
 *
 * \param[in]  sampleData       A linearised version of the sample points (expressed relative to the current bounding box).
 * \param[in]  sampleCount      The total number of samples (including those that fall outside the bounds of the frame).
 * \param[out] keepSamples      An output array into which to write whether or not each sample bounding box is to be retained.
 * \param[in]  bbX              The x coordinate of the current bounding box's anchor point.
 * \param[in]  bbY              The y coordinate of the current bounding box's anchor point.
 * \param[in]  bbWidth          The width of the current bounding box.
 * \param[in]  bbHeight         The height of the current bounding box.
 * \param[in]  frameWidth       The width of the frame window.
 * \param[in]  frameHeight      The height of the frame window.
 * \param[in]  threadsPerBlock  The number of threads to use in each CUDA block.
 */
extern "C" void filter_samples(float *sampleData, size_t sampleCount, int *keepSamples,
                               float bbX, float bbY, float bbWidth, float bbHeight,
                               int frameWidth, int frameHeight, int threadsPerBlock = 16);

}

#endif
