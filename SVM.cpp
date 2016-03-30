/**
 * thunderstruck/tracker: SVM.cpp
 */
#include "stdafx.h"
#include "SVM.h"

#include <stdexcept>

#include <boost/lexical_cast.hpp>

#include "SampleFilterer.h"
#include "Sampler.h"
#include "SVMImpl.h"
#include "GeomUtil.h"
#include "Timing.h"

namespace thunderstruck {

//#################### CONSTRUCTORS ####################

SVM::SVM(double C, const cv::Rect_<float>& initialBoundingBox, const FeatureCalculator_CPtr& featureCalculator,
         size_t nO, size_t nR, size_t maxSupportVectors)
: m_featureCalculator(featureCalculator),
  m_C(C), m_nO(nO), m_nR(nR),
  m_supportPatternToVectorMap(maxSupportVectors)
{
  // Determine the sample points for learning. Note that the
  // points are specified relative to the origin, which allows
  // us to reuse the same points from one frame to the next.
  const int angularSegments = 16;
  const float learningRadius = 60.0f;
  const int radialSegments = 5;
  std::vector<cv::Vec2f> learningSamples = Sampler::make_radial_samples(learningRadius, radialSegments, angularSegments);
  m_learningSampleCount = learningSamples.size();
  
  // Transfer the sample points across to the GPU.
  m_learningSampleData = Sampler::cpu_to_gpu(learningSamples);

  // Calculate the loss values for each sample.
  std::vector<double> lossValues(m_learningSampleCount);
  for(size_t i = 0; i < m_learningSampleCount; ++i)
  {
    float x = learningSamples[i][0];
    float y = learningSamples[i][1];
    cv::Rect_<float> boundingBox(initialBoundingBox.x + x, initialBoundingBox.y + y, initialBoundingBox.width, initialBoundingBox.height);
    lossValues[i] = loss_value(initialBoundingBox, boundingBox);
  }

  // Transfer the loss values across to the GPU.
  m_lossValues = make_gpu_vector(lossValues);

  // Create the GPU chunk vectors needed to hold the data for the support patterns.
  size_t maxSupportPatterns = maxSupportVectors; // there must be at least one support vector for each support pattern, so this is an upper bound
  m_features = GPUChunkVector<double>(maxSupportPatterns, m_learningSampleCount * m_featureCalculator->feature_count());
  m_keepSamples = GPUChunkVector<int>(maxSupportPatterns, m_learningSampleCount);

  // Create the vectors needed to hold the data for the support vectors.
  m_supportVectorsCPU.resize(maxSupportVectors, -1);
  m_supportVectorsGPU = GPUVector<int>(maxSupportVectors, -1);
  m_betasCPU.resize(maxSupportVectors, 0.0);
  m_betasGPU = GPUVector<double>(maxSupportVectors, 0.0);
  m_gradientsCPU.resize(maxSupportVectors, 0.0);
  m_gradientsGPU = GPUVector<double>(maxSupportVectors, 0.0);
  m_kernelMatrixCPU.resize(maxSupportVectors * maxSupportVectors, 0.0);
  m_kernelMatrixGPU = GPUVector<double>(maxSupportVectors * maxSupportVectors, 0.0);

  // Create temporary GPU vectors needed during SVM updating and weight calculation.
  m_gradientsForMinimisation = GPUVector<double>(m_learningSampleCount);
  m_weights = GPUVector<double>(m_featureCalculator->feature_count());
}
  
//#################### PUBLIC MEMBER FUNCTIONS ####################

void SVM::evaluate(const GPUVector<double>& sampleFeatures, const GPUVector<int>& keepSamples, const GPUVector<double>& sampleResults) const
{
  // Evaluate the SVM on the samples.
#if 1
  CUDA_TIME({
    calculate_svm_weights(
      m_supportVectorsGPU.get(), m_betasGPU.get(), max_support_vectors(),
      m_features.get(), m_featureCalculator->feature_count(), m_weights.get()
    );
    evaluate_svm_linear(m_weights.get(), m_featureCalculator->feature_count(), sampleFeatures.get(), keepSamples.get(), keepSamples.size(), sampleResults.get());
  }, microseconds);
#else
  CUDA_TIME({
    evaluate_svm_gaussian(
      m_supportVectorsGPU.get(), m_betasGPU.get(), max_support_vectors(),
      m_features.get(), m_featureCalculator->feature_count(),
      sampleFeatures.get(), keepSamples.get(), keepSamples.size(),
      sampleResults.get(), 0.2
    );
  }, microseconds);
#endif
  std::cout << "---- Time to evaluate SVM: " << dur << '\n';
}

size_t SVM::max_support_vectors() const
{
  return m_supportVectorsCPU.size();
}

void SVM::update(const GPUTexture1b& frame, const cv::Rect_<float>& curBoundingBox, const boost::optional<GPUTexture1i>& integralFrame)
{
  // Allocate a support pattern index.
  size_t index = m_supportPatternIDAllocator.allocate();

  // Get the (uninitialised) feature and sample flag vectors on the GPU for the allocated pattern index.
  GPUVector<double> features = m_features.get_chunk(index);
  GPUVector<int> keepSamples = m_keepSamples.get_chunk(index);

  // Determine which of the new samples are entirely within the current frame.
  {
    CUDA_TIME(filter_samples(
      m_learningSampleData.get(), m_learningSampleCount, keepSamples.get(),
      curBoundingBox.x, curBoundingBox.y, curBoundingBox.width, curBoundingBox.height,
      frame.width(), frame.height()
    ), microseconds);
    std::cout << "Time to filter samples for learning: " << dur << '\n';
  }

  // Calculate the feature vectors for those new samples.
  {
    CUDA_TIME(m_featureCalculator->calculate_features(
      frame.get(), m_learningSampleData.get(), keepSamples.get(), m_learningSampleCount,
      curBoundingBox, features.get(), integralFrame ? integralFrame->get() : 0
    ), microseconds);
    std::cout << "Time to calculate features for learning: " << dur << '\n';
  }

  // Run the update steps for the SVM, making sure that the budget for support vectors is not exceeded.
  {
    CUDA_TIME(process_new(index), microseconds);
    std::cout << "Time to run ProcessNew: " << dur << '\n';
  }

  for(size_t i = 0; i < m_nR; ++i)
  {
    reprocess();
  }
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

size_t SVM::add_support_vector(size_t i, size_t outputIndex)
{
  // Calculate the gradient value for the support vector.
  // TODO: This is messy - there must be a better way.
  size_t featureCount = m_featureCalculator->feature_count();
  GPUVector<double> sampleFeatures(m_features.get_chunk(i).get() + outputIndex * featureCount, featureCount);
  GPUVector<int> keepSamples(1, 1);
  GPUVector<double> gradients(1);
  calculate_gradients(sampleFeatures, keepSamples, gradients);
  double gradient = gradients.get(0);

  return add_support_vector(i, outputIndex, gradient);
}

size_t SVM::add_support_vector(size_t i, size_t outputIndex, double gradient)
{
  // Add the support vector and update the kernel matrix appropriately.
  size_t svIndex = m_supportVectorIDAllocator.allocate();
  int svRef = i * m_learningSampleCount + outputIndex;
  m_supportVectorsCPU[svIndex] = svRef;
  m_supportVectorsGPU.set(svIndex, svRef);
  m_betasCPU[svIndex] = 0.0;
  m_betasGPU.set(svIndex, 0.0);
  m_gradientsCPU[svIndex] = gradient;
  m_gradientsGPU.set(svIndex, gradient);
  update_kernel_matrix(m_kernelMatrixGPU.get(), svIndex, m_supportVectorsGPU.size(), m_supportVectorsGPU.get(), m_featureCalculator->feature_count(), m_features.get());
  m_kernelMatrixCPU = m_kernelMatrixGPU.to_cpu();

  // Record the association between the support vector and its support pattern (this is needed to control support pattern removal).
  m_supportPatternToVectorMap[i].insert(svIndex);
  m_supportVectorToPatternMap.insert(std::make_pair(svIndex, i));

  return svIndex;
}

void SVM::calculate_gradients(const GPUVector<double>& sampleFeatures, const GPUVector<int>& keepSamples, const GPUVector<double>& gradients) const
{
  GPUVector<double> evaluationResults(keepSamples.size());
  evaluate(sampleFeatures, keepSamples, evaluationResults);
  calculate_svm_gradients(m_lossValues.get(), evaluationResults.get(), keepSamples.get(), keepSamples.size(), gradients.get());
}

double SVM::kernel_value(size_t i, size_t j) const
{
  return m_kernelMatrixCPU[i * m_supportVectorsCPU.size() + j];
}

double SVM::loss_value(const cv::Rect_<float>& bb1, const cv::Rect_<float>& bb2) const
{
  return 1.0 - GeomUtil::compute_overlap(bb1, bb2);
}

void SVM::maintain_budget(const boost::optional<size_t>& protectedSvIndex)
{
  // If we've reached the maximum number of support vectors that are allowed, remove the least useful one,
  // namely a negative support vector whose removal has least effect on the SVM's weight vector.
  if(m_supportVectorIDAllocator.used_count() == m_supportVectorsCPU.size())
  {
    double bestDeltaSquared = INT_MAX;
    size_t bestMinusIndex = INT_MAX;
    size_t bestPlusIndex = INT_MAX;

    const std::set<int>& used = m_supportVectorIDAllocator.used();
    for(std::set<int>::const_iterator it = used.begin(), iend = used.end(); it != iend; ++it)
    {
      size_t minusIndex = *it;
      if(output_index(minusIndex) != 0)
      {
        // Find the corresponding positive support vector.
        size_t patternIndex = m_supportVectorToPatternMap[minusIndex];
        const std::set<size_t>& svs = m_supportPatternToVectorMap[patternIndex];
        size_t plusIndex = INT_MAX;
        for(std::set<size_t>::const_iterator jt = svs.begin(), jend = svs.end(); jt != jend; ++jt)
        {
          if(output_index(*jt) == 0)
          {
            plusIndex = *jt;
            break;
          }
        }

        // Calculate the square of the change in the weight vector that would be caused by removing the negative support vector.
        double b_r_y = m_betasCPU[minusIndex];
        double deltaSquared = b_r_y * b_r_y * (kernel_value(minusIndex, minusIndex) + kernel_value(plusIndex, plusIndex) - 2 * kernel_value(plusIndex, minusIndex));

        // If it's less than the current best and neither the negative or positive support vectors is protected,
        // make the negative support vector the new candidate for removal.
        if(deltaSquared < bestDeltaSquared &&
           !(protectedSvIndex && (minusIndex == *protectedSvIndex || plusIndex == *protectedSvIndex)))
        {
          bestDeltaSquared = deltaSquared;
          bestMinusIndex = minusIndex;
          bestPlusIndex = plusIndex;
        }
      }
    }

    // Adjust the beta value of the corresponding positive support vector to compensate for the negative one we're removing.
    m_betasCPU[bestPlusIndex] += m_betasCPU[bestMinusIndex];
    m_betasGPU.set(bestPlusIndex, m_betasCPU[bestPlusIndex]);

    // Remove the negative support vector.
    remove_support_vector(bestMinusIndex);

    // If the corresponding positive support vector is no longer useful, remove that as well.
    if(m_betasCPU[bestPlusIndex] < 1e-8)
    {
      remove_support_vector(bestPlusIndex);
    }

    // Update the gradient values of all the support vectors.
    // FIXME: This is messy and inefficient.
    for(std::set<int>::const_iterator it = used.begin(), iend = used.end(); it != iend; ++it)
    {
      size_t patternIndex = m_supportVectorToPatternMap.find(*it)->second;
      size_t outputIndex = output_index(*it);
      size_t featureCount = m_featureCalculator->feature_count();
      GPUVector<double> sampleFeatures(m_features.get_chunk(patternIndex).get() + outputIndex * featureCount, featureCount);
      GPUVector<int> keepSamples(1, 1);
      GPUVector<double> gradients(1);
      calculate_gradients(sampleFeatures, keepSamples, gradients);
      double gradient = gradients.get(0);
      m_gradientsCPU[*it] = gradient;
      m_gradientsGPU.set(*it, gradient);
    }
  }
}

size_t SVM::maximise_gradient_among_support_vectors_with_constraint(size_t i) const
{
  double bestGradient = INT_MIN;
  size_t bestIndex = INT_MAX;

  const std::set<size_t>& svs = m_supportPatternToVectorMap[i];
  for(std::set<size_t>::const_iterator jt = svs.begin(), jend = svs.end(); jt != jend; ++jt)
  {
    size_t j = *jt;
    double beta = m_betasCPU[j];
    double gradient = m_gradientsCPU[j];
    if(gradient > bestGradient && beta < m_C * (output_index(j) == 0 ? 1 : 0))
    {
      bestGradient = gradient;
      bestIndex = *jt;
    }
  }

  return bestIndex;
}

std::pair<size_t,double> SVM::minimise_gradient(size_t i) const
{
  // Calculate the gradients for every sample associated with this support pattern.
  calculate_gradients(m_features.get_chunk(i), m_keepSamples.get_chunk(i), m_gradientsForMinimisation);

  // Find a smallest gradient.
  std::vector<double> gradientsForMinimisation = m_gradientsForMinimisation.to_cpu();
  double bestGradient = INT_MAX;
  size_t bestOutputIndex = INT_MAX;
  for(size_t j = 1, size = gradientsForMinimisation.size(); j < size; ++j)
  {
    if(gradientsForMinimisation[j] < bestGradient)
    {
      bestGradient = gradientsForMinimisation[j];
      bestOutputIndex = j;
    }
  }

  return std::make_pair(bestOutputIndex, bestGradient);
}

size_t SVM::minimise_gradient_among_support_vectors(size_t i) const
{
  double bestGradient = INT_MAX;
  size_t bestIndex = INT_MAX;

  const std::set<size_t>& svs = m_supportPatternToVectorMap[i];
  for(std::set<size_t>::const_iterator jt = svs.begin(), jend = svs.end(); jt != jend; ++jt)
  {
    size_t j = *jt;
    double gradient = m_gradientsCPU[j];
    if(gradient < bestGradient)
    {
      bestGradient = gradient;
      bestIndex = *jt;
    }
  }

  return bestIndex;
}

void SVM::optimize()
{
  size_t i = random_support_pattern();
  if(i == INT_MAX) return;
  size_t plusIndex = maximise_gradient_among_support_vectors_with_constraint(i);
  if(plusIndex == INT_MAX) return;
  size_t minusIndex = minimise_gradient_among_support_vectors(i);
  smo_step(plusIndex, minusIndex);
}

size_t SVM::output_index(size_t svIndex) const
{
  return m_supportVectorsCPU[svIndex] % m_learningSampleCount;
}

void SVM::process_new(size_t i)
{
  maintain_budget();

  size_t plusIndex = add_support_vector(i, 0);
  std::pair<size_t,double> minGradResult = minimise_gradient(i);

  maintain_budget(plusIndex);

  size_t minusIndex = add_support_vector(i, minGradResult.first, minGradResult.second);
  smo_step(plusIndex, minusIndex);
}

void SVM::process_old()
{
  if(m_supportPatternIDAllocator.used_count() == 0) return;

  // Choose the support pattern to optimise.
  size_t i = random_support_pattern();
  if(i == INT_MAX) return;

  // Pick y+ as the y in one of the pattern's support vectors with maximum gradient (subject to constraints).
  size_t plusIndex = maximise_gradient_among_support_vectors_with_constraint(i);
  if(plusIndex == INT_MAX) return;

  // Pick y- as the y for the support pattern with minimum gradient.
  std::pair<size_t,double> minGradResult = minimise_gradient(i);

  // If there is already a support vector corresponding to y-, use it.
  const std::set<size_t>& svs = m_supportPatternToVectorMap[i];
  size_t minusIndex = INT_MAX;
  for(std::set<size_t>::const_iterator jt = svs.begin(), jend = svs.end(); jt != jend; ++jt)
  {
    if(output_index(*jt) == minGradResult.first)
    {
      minusIndex = *jt;
      break;
    }
  }

  // If not, add a new support vector.
  if(minusIndex == INT_MAX)
  {
    maintain_budget(plusIndex);
    minusIndex = add_support_vector(i, minGradResult.first, minGradResult.second);
  }

  // Perform the actual optimization.
  smo_step(plusIndex, minusIndex);
}

size_t SVM::random_support_pattern() const
{
  const std::set<int>& used = m_supportPatternIDAllocator.used();
  if(used.empty()) return INT_MAX;
  std::set<int>::const_iterator it = used.begin();
  std::advance(it, rand() % used.size());
  return *it;
}

void SVM::remove_support_vector(size_t svIndex)
{
  m_supportVectorIDAllocator.deallocate(svIndex);
  m_supportVectorsCPU[svIndex] = -1;
  m_supportVectorsGPU.set(svIndex, -1);

  // Update the association between the support vector and its pattern.
  std::map<size_t,size_t>::iterator it = m_supportVectorToPatternMap.find(svIndex);
  if(it == m_supportVectorToPatternMap.end())
  {
    throw std::runtime_error("Unknown support vector: " + boost::lexical_cast<std::string>(svIndex));
  }

  size_t patternIndex = it->second;
  m_supportVectorToPatternMap.erase(it);

  std::set<size_t>& supportVectorsForPattern = m_supportPatternToVectorMap[patternIndex];
  supportVectorsForPattern.erase(svIndex);

  // If this is the last support vector for this pattern, remove the pattern as well.
  if(supportVectorsForPattern.empty())
  {
    m_supportPatternIDAllocator.deallocate(patternIndex);
  }
}

void SVM::reprocess()
{
  {
    CUDA_TIME(process_old(), microseconds);
    std::cout << "Time to run ProcessOld: " << dur << '\n';
  }

  CUDA_TIME(
    for(size_t i = 0; i < m_nO; ++i)
    {
      optimize();
    },
  microseconds);
  std::cout << "Time to run Optimizes: " << dur << '\n';
}

void SVM::smo_step(size_t plusIndex, size_t minusIndex)
{
  // Ensure that we never try to optimise a support vector against itself.
  if(plusIndex == minusIndex)
  {
    return;
  }

  // Step 1: Retrieve b_i(y+) and g_i(y+).
  double& b_i_yPlus = m_betasCPU[plusIndex];
  double& g_i_yPlus = m_gradientsCPU[plusIndex];

  // Step 2: Retrieve b_i(y-) and g_i(y-).
  double& b_i_yMinus = m_betasCPU[minusIndex];
  double& g_i_yMinus = m_gradientsCPU[minusIndex];

  // Note: If g_i(y+) is not larger than g_i(y-) by a small amount then lambda will be zero.
  if(g_i_yPlus - g_i_yMinus >= 1e-5)
  {
    // Step 3: Calculate lambda.
    double k00 = kernel_value(plusIndex, plusIndex);
    double k11 = kernel_value(minusIndex, minusIndex);
    double k01 = kernel_value(plusIndex, minusIndex);
    double lambda = (g_i_yPlus - g_i_yMinus) / (k00 + k11 - 2 * k01);

    // Step 4: Constrain lambda.
    if(lambda < 0.0) lambda = 0.0;

    // Note: The positive output for a support pattern is always at index 0.
    double maxLambda = m_C * (output_index(plusIndex) == 0 ? 1 : 0) - b_i_yPlus;
    if(lambda > maxLambda) lambda = maxLambda;

    // Step 5: Update the beta values for the relevant support vectors.
    b_i_yPlus += lambda;
    b_i_yMinus -= lambda;
    m_betasGPU.set(plusIndex, b_i_yPlus);
    m_betasGPU.set(minusIndex, b_i_yMinus);

    // Step 6: Update the gradient values for all support vectors.
    update_gradient_values(lambda, plusIndex, minusIndex, m_supportVectorsGPU.get(), m_gradientsGPU.get(), m_supportVectorsGPU.size(), m_kernelMatrixGPU.get());
    m_gradientsCPU = m_gradientsGPU.to_cpu();
  }

  // Step 7: Update the support vector set (if either b_i_y+ or b_i_y- is too small, we remove the corresponding support vector).
  const double EPSILON = 1e-8;
  if(fabs(b_i_yPlus) < EPSILON) remove_support_vector(plusIndex);
  if(fabs(b_i_yMinus) < EPSILON) remove_support_vector(minusIndex);
}

}
