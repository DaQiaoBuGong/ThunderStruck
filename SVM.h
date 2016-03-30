/**
 * thunderstruck/tracker: SVM.h
 */

#ifndef H_THUNDERSTRUCK_SVM
#define H_THUNDERSTRUCK_SVM

#include <boost/optional.hpp>

#include "FeatureCalculator.h"
#include "GPUChunkVector.h"
#include "GPUTexture1.h"
#include "IDAllocator.h"

namespace thunderstruck {

/**
 * \brief An instance of this class represents the type of support vector machine used by a Struck tracker.
 */
class SVM
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The feature calculator to use. */
  FeatureCalculator_CPtr m_featureCalculator;

  /** A GPU vector to hold the gradients computed during an invocation of minimise_gradient. We create it once rather than on each invocation to avoid overhead. */
  GPUVector<double> m_gradientsForMinimisation;

  /** The number of sample points for learning. */
  size_t m_learningSampleCount;

  /** A GPU vector containing a linearised version of the sample points for learning. */
  GPUVector<float> m_learningSampleData;

  /** The loss values for the learning samples. */
  GPUVector<double> m_lossValues;

  /** A GPU vector into which to write the weights of the SVM when they are being calculated. */
  GPUVector<double> m_weights;

  //~~~~~~~~~~~~~~~~~~~~ SVM PARAMETERS ~~~~~~~~~~~~~~~~~~~~

  /** The C value to use (this is a standard SVM parameter controlling the slack constraints). */
  double m_C;

  /** The number of Optimize steps to take for each ProcessOld step. */
  size_t m_nO;

  /** The number of Reprocess steps to take for each ProcessNew step. */
  size_t m_nR;

  //~~~~~~~~~~~~~~~~~~~~ SUPPORT PATTERNS ~~~~~~~~~~~~~~~~~~~~

  /** A GPU vector containing the feature vectors for each potential support vector in each support pattern. */
  GPUChunkVector<double> m_features;

  /** A GPU vector specifying which samples are valid for each support pattern. */
  GPUChunkVector<int> m_keepSamples;

  /** The ID allocator for support patterns. */
  IDAllocator m_supportPatternIDAllocator;

  /** A map from support pattern IDs to vector IDs. */
  std::vector<std::set<size_t> > m_supportPatternToVectorMap;

  //~~~~~~~~~~~~~~~~~~~~ SUPPORT VECTORS ~~~~~~~~~~~~~~~~~~~~

  /** The beta values for the currently-active support vectors (CPU version). */
  std::vector<double> m_betasCPU;

  /** The beta values for the currently-active support vectors (GPU version). */
  GPUVector<double> m_betasGPU;

  /** The gradient values for the currently-active support vectors (CPU version). */
  std::vector<double> m_gradientsCPU;

  /** The gradient values for the currently-active support vectors (GPU version). */
  GPUVector<double> m_gradientsGPU;

  /** A matrix of pairwise kernel values for the currently-active support vectors (CPU version). */
  std::vector<double> m_kernelMatrixCPU;

  /** A matrix of pairwise kernel values for the currently-active support vectors (GPU version). */
  GPUVector<double> m_kernelMatrixGPU;

  /** A CPU vector containing references to the currently-active support vectors (-1 indicates that an element does not currently reference a support vector). */
  std::vector<int> m_supportVectorsCPU;

  /** A GPU vector containing references to the currently-active support vectors (-1 indicates that an element does not currently reference a support vector). */
  GPUVector<int> m_supportVectorsGPU;

  /** The ID allocator for the support vector chunk array. */
  IDAllocator m_supportVectorIDAllocator;

  /** A map from support vector IDs to pattern IDs. */
  std::map<size_t,size_t> m_supportVectorToPatternMap;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs an SVM.
   *
   * \param[in] C                   The C value to use (this is a standard SVM parameter controlling the slack constraints).
   * \param[in] initialBoundingBox  The initial bounding box around the object being tracked.
   * \param[in] featureCalculator   The feature calculator to use.
   * \param[in] nO                  The number of Optimize steps to take for each ProcessOld step.
   * \param[in] nR                  The number of Reprocess steps to take for each ProcessNew step.
   * \param[in] maxSupportVectors   The maximum number of support vectors that the SVM should maintain.
   */
  explicit SVM(double C, const cv::Rect_<float>& initialBoundingBox, const FeatureCalculator_CPtr& featureCalculator,
               size_t nO = 10, size_t nR = 10, size_t maxSupportVectors = 100);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Evaluates the SVM on a set of samples.
   *
   * \param[in]  sampleFeatures The feature vectors for the samples.
   * \param[in]  keepSamples    Which of the samples are valid.
   * \param[out] sampleResults  An array into which to store the evaluation results for the samples.
   */
  void evaluate(const GPUVector<double>& sampleFeatures, const GPUVector<int>& keepSamples, const GPUVector<double>& sampleResults) const;

  /**
   * \brief Gets the maximum number of support vectors that the SVM should maintain.
   *
   * \return  The maximum number of support vectors that the SVM should maintain.
   */
  size_t max_support_vectors() const;

  /**
   * \brief Updates the SVM with data from a new frame.
   *
   * \param[in] frame         The frame with which to update the SVM.
   * \param[in] boundingBox   The bounding box of the tracked object in the frame.
   * \param[in] integralFrame The integral image for the frame, if needed by the feature calculator.
   */
  void update(const GPUTexture1b& frame, const cv::Rect_<float>& boundingBox, const boost::optional<GPUTexture1i>& integralFrame);

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Adds a new support vector.
   *
   * \param[in] i           The index of the support pattern to use.
   * \param[in] outputIndex The index of the output (within that support pattern) to use.
   * \return                The index of the new support vector in the support vectors array.
   */
  size_t add_support_vector(size_t i, size_t outputIndex);

  /**
   * \brief Adds a new support vector.
   *
   * \param[in] i           The index of the support pattern to use.
   * \param[in] outputIndex The index of the output (within that support pattern) to use.
   * \param[in] gradient    The gradient of the new support vector.
   * \return                The index of the new support vector in the support vectors array.
   */
  size_t add_support_vector(size_t i, size_t outputIndex, double gradient);

  /**
   * \brief Calculates the gradient values for a set of samples.
   *
   * \param[in]  sampleFeatures The feature vectors for the samples.
   * \param[in]  keepSamples    Which of the samples are valid.
   * \param[out] gradients      An array into which to store the gradients.
   */
  void calculate_gradients(const GPUVector<double>& sampleFeatures, const GPUVector<int>& keepSamples, const GPUVector<double>& gradients) const;

  /**
   * \brief Looks up the value of the kernel for the specified pair of support vectors.
   *
   * \param[in] i The index of the first support vector.
   * \param[in] j The index of the second support vector.
   * \return      The value of the kernel for the specified support vectors.
   */
  double kernel_value(size_t i, size_t j) const;

  /**
   * \brief Computes the value of the loss function for two bounding boxes.
   *
   * \param[in] bb1 The first bounding box.
   * \param[in] bb2 The second bounding box.
   * \return        The loss value for the two bounding boxes.
   */
  double loss_value(const cv::Rect_<float>& bb1, const cv::Rect_<float>& bb2) const;

  /**
   * \brief TODO
   */
  void maintain_budget(const boost::optional<size_t>& protectedSvIndex = boost::none);

  /**
   * \brief Finds the index of the support vector associated with pattern i that maximises the gradient g_i(y)
   *        subject to the constraint b_i(y) < delta(y,y_i)C, where y is the output associated with the support
   *        vector.
   *
   * \param[in] The index of the pattern whose support vectors are to be searched.
   * \return    The index of the support vector associated with the pattern that maximises the gradient, subject to the constraint.
   */
  size_t maximise_gradient_among_support_vectors_with_constraint(size_t i) const;

  /**
   * \brief Finds the (negative) output y associated with support pattern i that minimises the gradient g_i(y),
   *        together with the associated gradient value.
   *
   * \param[in] i The index of the support pattern whose outputs are to be searched.
   * \return      A pair, the first component of which is the index of the output within the support pattern
   *              that minimises the gradient, and the second component of which is the value of the gradient
   *              for that output.
   */
  std::pair<size_t,double> minimise_gradient(size_t i) const;

  /**
   * \brief Finds the index of the support vector associated with pattern i that minimises the gradient g_i(y),
   *        where y is the output associated with the support vector. Note that this index is into the list of
   *        support vectors, *not* into the support pattern's list of outputs.
   *
   * \param[in] i The index of the pattern whose support vectors are to be searched.
   * \return      The index of the support vector associated with the pattern that minimises the gradient.
   */
  size_t minimise_gradient_among_support_vectors(size_t i) const;

  /**
   * \brief Performs an Optimize step as defined in the original paper.
   *
   * This performs an SMO step, choosing a random support pattern x_i and then picking y+ as the y among
   * all support vectors for that pattern that has the maximum gradient (subject to constraints) and y-
   * as the y among all support vectors for that pattern with minimum gradient. Note that finding y+ and
   * y- is much quicker here than in the ProcessOld case, since we only need to search the current support
   * vectors for the chosen pattern rather than the entire output space.
   */
  void optimize();

  /**
   * \brief Gets the output index associated with the specified support vector.
   *
   * \param[in] svIndex The index of the support vector.
   * \return            The output index associated with the support vector.
   */
  size_t output_index(size_t svIndex) const;

  /**
   * \brief Performs a ProcessNew step as defined in the original paper.
   *
   * This performs an SMO step, using the y associated with the support pattern (i.e. y_i) as y+
   * and the y with minimum gradient as y-.
   *
   * \param[in] i The index of the support pattern on which to perform the step.
   */
  void process_new(size_t i);

  /**
   * \brief Performs a ProcessOld step as defined in the original paper.
   *
   * This performs an SMO step, choosing a random support pattern x_i and then picking y+ as the y
   * with maximum gradient (subject to constraints) and y- as the y with minimum gradient.
   */
  void process_old();

  /**
   * \brief Picks a random support pattern.
   *
   * \return  The index of the chosen support pattern.
   */
  size_t random_support_pattern() const;

  /**
   * \brief Removes the specified support vector.
   *
   * Note that this can also result in the corresponding support pattern being removed in due course
   * if there are no more support vectors associated with it.
   *
   * \param[in] svIndex The index of the support vector to remove.
   */
  void remove_support_vector(size_t svIndex);

  /**
   * \brief Performs a Reprocess step, i.e. a ProcessOld step followed by a fixed number of Optimize steps.
   */
  void reprocess();

  /**
   * \brief Performs a sequential minimal optimisation (SMO) step on the selected support vectors and updates the SVM.
   *
   * \param[in] plusIndex   The index indicating the + support vector to optimise.
   * \param[in] minusIndex  The index indicating the - support vector to optimise.
   */
  void smo_step(size_t plusIndex, size_t minusIndex);
};

}

#endif
