#include <Restructor/cuda/include/cudaTypeDef.cuh>

namespace RestructorType {
    namespace cudaFunc {
        __global__ void matchAndTriangulateColor_CUDA(cv::cuda::PtrStep<float> leftImg, cv::cuda::PtrStep<float> rightImg, cv::cuda::PtrStep<cv::Vec3b> colorImg, const int rows, const int cols,
            const int minDisparity, const int maxDisparity, const float minDepth, const float maxDepth, const Eigen::Matrix4f Q,
            const Eigen::Matrix3f M3, const Eigen::Matrix3f R, const Eigen::Vector3f T, const Eigen::Matrix3f R1_inv, cv::cuda::PtrStep<uint16_t> mapDepth, cv::cuda::PtrStep<cv::Vec3b> mapColor) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x < cols && y < rows) {
                if (0 >= leftImg.ptr(y)[x]) {
                    mapDepth.ptr(y)[x] = 0;
                    return;
                }
                const float f = Q(2,3);
                const float tx = -1.0 / Q(3,2);
                const float cxlr = Q(3,3) * tx;
                const float cx = -1.0 * Q(0,2);
                const float cy = -1.0 * Q(1,3);
                const float threshod = 0.1;
                float cost = 0;
                int k = 0;
                float minCost = FLT_MAX;
                for (int d = minDisparity; d < maxDisparity; d++) {
                    if (x - d <0 || x - d >cols - 1) {
                        mapDepth.ptr(y)[x] = 0;
                        continue;
                    }
                    cost = cuda::std::abs(leftImg.ptr(y)[x] - rightImg.ptr(y)[x - d]);
                    if (cost < minCost) {
                        minCost = cost;
                        k = d;
                    }
                }
                if (minCost > threshod || x - k + 1 > cols - 1 || x - k - 1 < 0) {
                    mapDepth.ptr(y)[x] = 0;
                    return;
                }
                float dived = rightImg.ptr(y)[x - k + 1] - rightImg.ptr(y)[x - k - 1];
                if (cuda::std::abs(dived) < 0.001) {
                    dived = 0.001;
                }
                float depth = k + 2 * (rightImg.ptr(y)[x - k] - leftImg.ptr(y)[x]) / dived;
                if (depth<minDisparity || depth>maxDisparity) {
                    mapDepth.ptr(y)[x] = 0;
                    return;
                }
                Eigen::Vector3f vertex;
                vertex(0,0) = -1.0f * tx * (x - cx) / (depth - cxlr);
                vertex(1,0) = -1.0f * tx * (y - cy) / (depth - cxlr);
                vertex(2,0) = -1.0f * tx * f / (depth - cxlr);
                const Eigen::Vector3f leftVertex = R1_inv * vertex;
                const float depthMapped = leftVertex(2,0);
                if (depthMapped < minDepth || depthMapped > maxDepth) {
                    mapDepth.ptr(y)[x] = 0;
                }
                const Eigen::Vector3f imgMapped= M3 * (R * leftVertex + T);
                const int x_maped = imgMapped(0,0) / imgMapped(2,0);
                const int y_maped = imgMapped(1,0) / imgMapped(2,0);
                if (x_maped < cols && y_maped < rows && 0 < x_maped && 0 < y_maped) {
                    mapDepth.ptr(y)[x] = depthMapped * 5;
                    mapColor.ptr(y)[x] = colorImg.ptr(y_maped)[x_maped];
                }
            }
        }

        void depthColorMap(const cv::cuda::GpuMat& leftImg_, const cv::cuda::GpuMat& rightImg_, const cv::cuda::GpuMat& colorImg_, const int rows, const int cols,
            const int minDisparity, const int maxDisparity, const float minDepth, const float maxDepth, const Eigen::Matrix4f& Q,
            const Eigen::Matrix3f& M3, const Eigen::Matrix3f& R, const Eigen::Vector3f& T, const Eigen::Matrix3f& R1_inv,
            cv::cuda::GpuMat& depthMap, cv::cuda::GpuMat& mapColor, const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            matchAndTriangulateColor_CUDA << <grid, block,0, stream >> > (leftImg_, rightImg_, colorImg_, rows, cols,
                minDisparity, maxDisparity, minDepth, maxDepth, Q,
                M3, R, T, R1_inv, depthMap, mapColor);
        }
    }
}

