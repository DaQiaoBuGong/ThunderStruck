/**
 * thunderstruck/tracker: SVMImpl.h
 */

#ifndef H_THUNDERSTRUCK_SVMIMPL
#define H_THUNDERSTRUCK_SVMIMPL

namespace thunderstruck {

//#################### WRAPPER FUNCTION DECLARATIONS ####################

/**
 * \brief Calculates the gradient values for the specified set of samples.
 *
 * \param[in]  lossValues         The loss values for the samples.
 * \param[in]  evaluationResults  The results of evaluating the SVM for each of the samples.
 * \param[in]  keepSamples        An array specifying which of the samples are valid.
 * \param[in]  sampleCount        The number of samples.
 * \param[out] gradients          An array into which to store the calculated gradients.
 * \param[in]  threadsPerBlock    The number of threads to use in each CUDA block.
 */
extern "C" void calculate_svm_gradients(double *lossValues, double *evaluationResults, int *keepSamples, size_t sampleCount, double *gradients, int threadsPerBlock = 256);

/**
 * \brief Calculates the weights for the SVM (assuming it is using a linear kernel).
 *
 * \param[in]  supportVectors     An array containing references to the currently-active support vectors in the SVM.
 * \param[in]  betas              An array containing the beta values for the currently-active support vectors in the SVM.
 * \param[in]  maxSupportVectors  The size of the support vector array.
 * \param[in]  features           An array containing the feature vectors for all potential support vectors in the SVM.
 * \param[in]  featureCount       The number of features in each feature vector.
 * \param[out] weights            An array into which to write the calculated weights.
 */
extern "C" void calculate_svm_weights(int *supportVectors, double *betas, size_t maxSupportVectors, double *features, size_t featureCount, double *weights);

/**
 * \brief Evaluates the SVM on the specified set of samples using a Gaussian kernel.
 *
 * \param[in]  supportVectors     An array containing references to the currently-active support vectors in the SVM.
 * \param[in]  betas              An array containing the beta values for the currently-active support vectors in the SVM.
 * \param[in]  maxSupportVectors  The size of the support vector array.
 * \param[in]  features           An array containing the feature vectors for all potential support vectors in the SVM.
 * \param[in]  featureCount       The number of features in each feature vector.
 * \param[in]  sampleFeatures     An array containing the feature vectors for the samples.
 * \param[in]  keepSamples        An array specifying which of the samples are valid.
 * \param[in]  sampleCount        The number of samples.
 * \param[out] sampleResults      An array into which to store the evaluation results for the samples.
 * \param[in]  sigma              The sigma value for the Gaussian kernel (which is evaluated as exp(-sigma * |x1 - x2|^2)).
 */
extern "C" void evaluate_svm_gaussian(int *supportVectors, double *betas, size_t maxSupportVectors,
                                      double *features, size_t featureCount,
                                      double *sampleFeatures, int *keepSamples, size_t sampleCount,
                                      double *sampleResults, double sigma = 0.2);

/**
 * \brief Evaluates the SVM on the specified set of samples using a linear kernel.
 *
 * \param[in]  weights        The weights for the SVM (assuming it is using a linear kernel).
 * \param[in]  featureCount   The number of features in each feature vector.
 * \param[in]  sampleFeatures An array containing the feature vectors for the samples.
 * \param[in]  keepSamples    An array specifying which of the samples are valid.
 * \param[in]  sampleCount    The number of samples.
 * \param[out] sampleResults  An array into which to store the evaluation results for the samples.
 */
extern "C" void evaluate_svm_linear(double *weights, size_t featureCount, double *sampleFeatures, int *keepSamples, size_t sampleCount, double *sampleResults);

/**
 * \brief Updates the gradient values for the support vectors at the end of an SMO step.
 *
 * \param[in] lambda            The lambda value used during the SMO step.
 * \param[in] plusIndex         The index of the + support vector optimised by the SMO step.
 * \param[in] minusIndex        The index of the - support vector optimised by the SMO step.
 * \param[in] supportVectors    An array containing references to the currently-active support vectors in the SVM.
 * \param[in] gradients         An array containing the gradient values for the currently-active support vectors in the SVM.
 * \param[in] maxSupportVectors The size of the support vector array.
 * \param[in] kernelMatrix      The kernel matrix.
 * \param[in] threadsPerBlock   The number of threads to use in each CUDA block.
 */
extern "C" void update_gradient_values(double lambda, size_t plusIndex, size_t minusIndex,
                                       int *supportVectors, double *gradients, size_t maxSupportVectors,
                                       double *kernelMatrix, int threadsPerBlock = 256);

/**
 * \brief Updates the kernel matrix entries for the i'th support vector.
 *
 * \param[in] kernelMatrix      The kernel matrix.
 * \param[in] i                 The index of the support vector whose entries need updating.
 * \param[in] maxSupportVectors The size of the support vector array.
 * \param[in] supportVectors    An array containing references to the currently-active support vectors in the SVM.
 * \param[in] featureCount      The number of features in each feature vector.
 * \param[in] features          An array containing the feature vectors for all potential support vectors in the SVM.
 * \param[in] threadsPerBlock   The number of threads to use in each CUDA block.
 */
extern "C" void update_kernel_matrix(double *kernelMatrix, size_t i, size_t maxSupportVectors, int *supportVectors, size_t featureCount, double *features,
                                     int threadsPerBlock = 256);

}

#endif
