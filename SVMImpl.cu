/**
 * thunderstruck/tracker: SVMImpl.cu
 */

#include "SVMImpl.h"
#include "cudaApi.cuh"

namespace thunderstruck {

//#################### CUDA FUNCTIONS ####################

__device__ double cf_gaussian_kernel(double *x1, double *x2, size_t featureCount, double sigma = 0.2)
{
  double squaredNorm = 0.0;
  for(size_t i = 0; i < featureCount; ++i)
  {
    double delta = x1[i] - x2[i];
    squaredNorm += delta * delta;
  }
  return exp(-sigma * squaredNorm);
}

__device__ double cf_linear_kernel(double *x1, double *x2, size_t featureCount)
{
  double result = 0.0;
  for(size_t i = 0; i < featureCount; ++i)
  {
    result += x1[i] * x2[i];
  }
  return result;
}

//#################### CUDA KERNELS ####################

__global__ void ck_calculate_gradients(double *lossValues, double *evaluationResults, int *keepSamples, size_t sampleCount, double *gradients)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < sampleCount)
  {
    gradients[tid] = keepSamples[tid] != 0 ? -lossValues[tid] - evaluationResults[tid] : 0.0;
  }
}

__global__ void ck_calculate_weights(int *supportVectors, double *betas, size_t maxSupportVectors, double *features, size_t featureCount, double *weights)
{
  // Note: These arrays are set to the largest sizes necessary to avoid the need to manage shared memory dynamically.
  __shared__ int sharedSupportVectors[200];
  __shared__ double sharedBetas[200];

  if(threadIdx.x < maxSupportVectors)
  {
    sharedSupportVectors[threadIdx.x] = supportVectors[threadIdx.x];
    sharedBetas[threadIdx.x] = betas[threadIdx.x];
  }

  __syncthreads();

  double weight = 0.0;

  for(int i = 0; i < maxSupportVectors; ++i)
  {
    int svRef = sharedSupportVectors[i];
    if(svRef != -1)
    {
      weight += sharedBetas[i] * features[svRef * featureCount + threadIdx.x];
    }
  }

  weights[threadIdx.x] = weight;
}

__global__ void ck_evaluate_svm_gaussian(int *supportVectors, double *betas, size_t maxSupportVectors,
                                         double *features, int featureCount,
                                         double *sampleFeatures, int *keepSamples, double *sampleResults,
                                         double sigma)
{
  __shared__ volatile double sharedResults[512]; // note: the volatile is crucial or the reduction may fail

  double sampleResult = 0.0;

  // 1 global read
  if(keepSamples[blockIdx.x] != 0)
  {
    // 1 global read
    double sampleFeature = sampleFeatures[blockIdx.x * featureCount + threadIdx.x];

    // Calculate the contribution from each support vector and add it to the final result.
    // To calculate the contribution from support vector i, the strategy is to calculate
    // (x[j] - x_i[j])^2 in each thread j, and then add up the results using reduction to
    // find (x - x_i)^2. We do the necessary exponentiation when adding to the final result.
    for(int i = 0; i < maxSupportVectors; ++i)
    {
      // 1 global read
      int svRef = supportVectors[i];
      if(svRef == -1) continue;

      // 1 global read
      // TODO: Pass in svRef * featureCount in an array?
      float delta = sampleFeature - features[svRef * featureCount + threadIdx.x];

      // 1 shared write
      sharedResults[threadIdx.x] = delta * delta;

      __syncthreads();

      // Perform a reduction to calculate the sum of the shared results, namely |x - x_i|^2.
      for(unsigned int s = featureCount / 2; s > 32; s >>= 1)
      {
        if(threadIdx.x < s)
        {
          sharedResults[threadIdx.x] += sharedResults[threadIdx.x + s];
        }
        __syncthreads();
      }

      if(threadIdx.x < 32)
      {
        sharedResults[threadIdx.x] += sharedResults[threadIdx.x + 32];
        sharedResults[threadIdx.x] += sharedResults[threadIdx.x + 16];
        sharedResults[threadIdx.x] += sharedResults[threadIdx.x + 8];
        sharedResults[threadIdx.x] += sharedResults[threadIdx.x + 4];
        sharedResults[threadIdx.x] += sharedResults[threadIdx.x + 2];
        sharedResults[threadIdx.x] += sharedResults[threadIdx.x + 1];
      }

      // Add the contribution from this support vector, namely beta_i * exp(-sigma * |x - x_i|^2), to the final result.
      if(threadIdx.x == 0)
      {
        // 1 global read, 1 shared read
        sampleResult += betas[i] * __expf(-sigma * sharedResults[0]);
        //sampleResult = __fma_rn(betas[i], __expf(-sigma * sharedResults[0]), sampleResult);
        //sampleResult = __fmaf_rn(betas[i], __expf(-sigma * sharedResults[0]), sampleResult);
      }
    }
  }

  if(threadIdx.x == 0)
  {
    // 1 global write
    sampleResults[blockIdx.x] = sampleResult;
  }
}

__global__ void ck_evaluate_svm_linear(double *weights, double *sampleFeatures, int *keepSamples, double *sampleResults)
{
  // Note: This array is set to the largest size necessary to avoid the need to manage shared memory dynamically.
  __shared__ double sharedFeatureResults[512];

  // Each thread block evaluates the SVM on a single sample, which is just a dot product of the form w . x, where x
  // contains the features for the sample. We use a thread for each feature, and then sum the results from the various
  // features at the end.

  double featureResult = 0.0;

  // 1 global read
  if(keepSamples[blockIdx.x] != 0)
  {
    // 2 global reads
    featureResult = weights[threadIdx.x] * sampleFeatures[blockIdx.x * blockDim.x + threadIdx.x];
  }

  // 1 shared write
  sharedFeatureResults[threadIdx.x] = featureResult;

  __syncthreads();

  // Sum the results from all the feature threads and write the final result for the sample to global memory.
  if(threadIdx.x == 0)
  {
    double sampleResult = 0.0;

    // blockDim.x (e.g. 256) shared reads
    for(int i = 0; i < blockDim.x; ++i)
    {
      sampleResult += sharedFeatureResults[i];
    }

    // 1 global write
    sampleResults[blockIdx.x] = sampleResult;
  }
}

__global__ void ck_update_gradient_values(double lambda, size_t plusIndex, size_t minusIndex,
                                          int *supportVectors, double *gradients, size_t maxSupportVectors,
                                          double *kernelMatrix)
{
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < maxSupportVectors && supportVectors[tid] != -1)
  {
    gradients[tid] -= lambda * (kernelMatrix[tid * maxSupportVectors + plusIndex] - kernelMatrix[tid * maxSupportVectors + minusIndex]);
  }
}

__global__ void ck_update_kernel_matrix(double *kernelMatrix, size_t i, size_t maxSupportVectors, int *supportVectors, size_t featureCount, double *features)
{
  size_t j = threadIdx.x + blockDim.x * blockIdx.x;
  if(j < maxSupportVectors)
  {
    int svJ = supportVectors[j];
    if(svJ != -1)
    {
      int svI = supportVectors[i];
      double *featuresI = features + svI * featureCount;
      double *featuresJ = features + svJ * featureCount;
#if 1
      double value = cf_linear_kernel(featuresI, featuresJ, featureCount);
#else
      double value = cf_gaussian_kernel(featuresI, featuresJ, featureCount);
#endif
      kernelMatrix[i * maxSupportVectors + j] = value;
      kernelMatrix[j * maxSupportVectors + i] = value;
    }
  }
}

//#################### WRAPPER FUNCTIONS ####################

void calculate_svm_gradients(double *lossValues, double *evaluationResults, int *keepSamples, size_t sampleCount, double *gradients, int threadsPerBlock)
{
  int numThreads = sampleCount;
  int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
  ck_calculate_gradients<<<numBlocks,threadsPerBlock>>>(lossValues, evaluationResults, keepSamples, sampleCount, gradients);
}

void calculate_svm_weights(int *supportVectors, double *betas, size_t maxSupportVectors, double *features, size_t featureCount, double *weights)
{
  ck_calculate_weights<<<1,featureCount>>>(supportVectors, betas, maxSupportVectors, features, featureCount, weights);
}

void evaluate_svm_gaussian(int *supportVectors, double *betas, size_t maxSupportVectors,
                           double *features, size_t featureCount,
                           double *sampleFeatures, int *keepSamples, size_t sampleCount,
                           double *sampleResults, double sigma)
{
  ck_evaluate_svm_gaussian<<<sampleCount,featureCount>>>(supportVectors, betas, maxSupportVectors,
                                                         features, featureCount,
                                                         sampleFeatures, keepSamples, sampleResults,
                                                         sigma);
}

void evaluate_svm_linear(double *weights, size_t featureCount, double *sampleFeatures, int *keepSamples, size_t sampleCount, double *sampleResults)
{
  ck_evaluate_svm_linear<<<sampleCount,featureCount>>>(weights, sampleFeatures, keepSamples, sampleResults);
}

void update_gradient_values(double lambda, size_t plusIndex, size_t minusIndex,
                            int *supportVectors, double *gradients, size_t maxSupportVectors,
                            double *kernelMatrix, int threadsPerBlock)
{
  int numThreads = maxSupportVectors;
  int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
  ck_update_gradient_values<<<numBlocks,threadsPerBlock>>>(lambda, plusIndex, minusIndex, supportVectors, gradients, maxSupportVectors, kernelMatrix);
}

void update_kernel_matrix(double *kernelMatrix, size_t i, size_t maxSupportVectors, int *supportVectors, size_t featureCount, double *features, int threadsPerBlock)
{
  // We need to update K_ij (for all j).
  int numThreads = maxSupportVectors;
  int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
  ck_update_kernel_matrix<<<numBlocks,threadsPerBlock>>>(kernelMatrix, i, maxSupportVectors, supportVectors, featureCount, features);
}

}
