#include <cuda/cudaTypeDef.cuh>

namespace sl {
    namespace wrapCreator {
        namespace cudaFunc {
            __global__ void wrapThreeStep(
                    const cv::cuda::PtrStep<uchar> firstStep,
                    const cv::cuda::PtrStep<uchar> secondStep,
                    const cv::cuda::PtrStep<uchar> thirdStep,
                    const int cols, const int rows,
                    cv::cuda::PtrStep<float> wrapImg,
                    cv::cuda::PtrStep<float> conditionImg) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    const float diffFT = firstStep.ptr(y)[x] - thirdStep.ptr(y)[x];
                    const float diffSSFT = 2.f * secondStep.ptr(y)[x] - firstStep.ptr(y)[x] - thirdStep.ptr(y)[x];
                    conditionImg.ptr(y)[x] = cuda::std::sqrtf(3.f * diffFT * diffFT + diffSSFT * diffSSFT) / 3.f;
                    if (conditionImg.ptr(y)[x] < 10.f)
                        wrapImg.ptr(y)[x] = -5.f;
                    else
                        wrapImg.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.f) * diffFT, diffSSFT);
                }
            }

            __global__ void wrapFourStep(
                    const cv::cuda::PtrStep<uchar> firstStep,
                    const cv::cuda::PtrStep<uchar> secondStep,
                    const cv::cuda::PtrStep<uchar> thirdStep,
                    const cv::cuda::PtrStep<uchar> fourthStep,
                    const int cols, const int rows,
                    cv::cuda::PtrStep<float> wrapImg,
                    cv::cuda::PtrStep<float> conditionImg) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    const float diffFS = fourthStep.ptr(y)[x] - secondStep.ptr(y)[x];
                    const float diffFT = firstStep.ptr(y)[x] - thirdStep.ptr(y)[x];
                    conditionImg.ptr(y)[x] = cuda::std::sqrtf(diffFS * diffFS + diffFT * diffFT) / 2.f;
                    if (conditionImg.ptr(y)[x] < 10.f)
                        wrapImg.ptr(y)[x] = -5.f;
                    else
                        wrapImg.ptr(y)[x] = cuda::std::atan2f(diffFS, diffFT);
                }
            }

            void getWrapImgSync(
                    const std::vector<cv::cuda::GpuMat> &imgs,
                    cv::cuda::GpuMat &wrapImg,
                    cv::cuda::GpuMat &conditionImg, const bool isCounter = false,
                    const cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                    const dim3 block = dim3(32, 8)) {
                const int stepNum = imgs.size();
                const int cols = imgs[0].cols;
                const int rows = imgs[0].rows;
                dim3 grid((cols + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
                cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(cvStream);
                switch (stepNum) {
                    case 3: {
                        wrapThreeStep<<<grid, block, 0, cudaStream>>>(
                                imgs[0], imgs[1], imgs[2],
                                cols, rows,
                                wrapImg, conditionImg);
                        break;
                    }
                    case 4: {
                        if (isCounter)
                            wrapFourStep<<<grid, block, 0, cudaStream>>>(
                                    imgs[2], imgs[3], imgs[0], imgs[1],
                                    cols, rows,
                                    wrapImg, conditionImg);
                        else
                            wrapFourStep<<<grid, block, 0, cudaStream>>>(
                                    imgs[0], imgs[1], imgs[2], imgs[3],
                                    cols, rows,
                                    wrapImg, conditionImg);
                        break;
                    }
                    default:
                        std::cout << "The " << stepNum << " step is not support in current!" << std::endl;
                        break;
                }
            }
        }// namespace cudaFunc
    }// namespace wrapCreator
}// namespace sl