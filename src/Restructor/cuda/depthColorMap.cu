#include <Restructor/cuda/include/cudaTypeDef.cuh>
#include <atomic>

namespace RestructorType {
    namespace cudaFunc {

        __global__ void matchAndTriangulateColor_CUDA(cv::cuda::PtrStep<float> leftImg, cv::cuda::PtrStep<float> rightImg, const int rows, const int cols,
            const int minDisparity, const int maxDisparity, const float minDepth, const float maxDepth, const Eigen::Matrix4f Q,
            const Eigen::Matrix3f M3, const Eigen::Matrix3f R, const Eigen::Vector3f T, const Eigen::Matrix3f R1_inv, cv::cuda::PtrStep<float> mapDepth) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x < cols && y < rows) {
                if (0 >= leftImg.ptr(y)[x]) {
                    return;
                }
                const float f = Q(2,3);
                const float tx = -1.0 / Q(3,2);
                const float cxlr = Q(3,3) * tx;
                const float cx = -1.0 * Q(0,3);
                const float cy = -1.0 * Q(1,3);
                const float threshod = 0.1;
                float cost = 0;
                int k = 0;
                float minCost = FLT_MAX;
                for (int d = minDisparity; d < maxDisparity; d++) {
                    if (x - d <0 || x - d >cols - 1) {
                        continue;
                    }
                    cost = cuda::std::abs(leftImg.ptr(y)[x] - rightImg.ptr(y)[x - d]);
                    if (cost < minCost) {
                        minCost = cost;
                        k = d;
                    }
                }
                if (minCost > threshod || x - k + 1 > cols - 1 || x - k - 1 < 0) {
                    return;
                }
                float dived = rightImg.ptr(y)[x - k + 1] - rightImg.ptr(y)[x - k - 1];
                if (cuda::std::abs(dived) < 0.001) {
                    dived = 0.001;
                }
                float disparity = k + 2 * (rightImg.ptr(y)[x - k] - leftImg.ptr(y)[x]) / dived;
                if (disparity<minDisparity || disparity>maxDisparity) {
                    return;
                }
                Eigen::Vector3f vertex;
                vertex(0,0) = -1.0f * tx * (x - cx) / (disparity - cxlr);
                vertex(1,0) = -1.0f * tx * (y - cy) / (disparity - cxlr);
                vertex(2,0) = -1.0f * tx * f / (disparity - cxlr);
                const Eigen::Vector3f colorVertex = R * (R1_inv * vertex) + T;
                const Eigen::Vector3f imgMapped= M3 * colorVertex;
                const int x_maped = imgMapped(0,0) / imgMapped(2,0);
                const int y_maped = imgMapped(1,0) / imgMapped(2,0);
                const float depthMaped = colorVertex(2, 0);
                if (x_maped < cols && y_maped < rows && 0 <= x_maped && 0 <= y_maped) {
                    if (depthMaped < minDepth || depthMaped > maxDepth) {
                        atomicExch(&mapDepth.ptr(y_maped)[x_maped], 0);
                        //mapDepth.ptr(y)[x] = 0;
                    }
                    atomicExch(&mapDepth.ptr(y_maped)[x_maped], depthMaped);
                    //mapDepth.ptr(y)[x] = depthMapped * 5;
                    //mapColor.ptr(y)[x] = colorImg.ptr(y_maped)[x_maped];
                }
            }
        }

        __global__ void matchAndTriangulate_CUDA(cv::cuda::PtrStep<float> leftImg, cv::cuda::PtrStep<float> rightImg, const int rows, const int cols,
            const int minDisparity, const int maxDisparity, const float minDepth, const float maxDepth, const Eigen::Matrix4f Q,
            const Eigen::Matrix3f M1, const Eigen::Matrix3f R1_inv, cv::cuda::PtrStep<float> mapDepth) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x < cols && y < rows) {
                if (0 >= leftImg.ptr(y)[x]) {
                    return;
                }
                const float f = Q(2, 3);
                const float tx = -1.0 / Q(3, 2);
                const float cxlr = Q(3, 3) * tx;
                const float cx = -1.0 * Q(0, 3);
                const float cy = -1.0 * Q(1, 3);
                const float threshod = 0.1;
                float cost = 0;
                int k = 0;
                float minCost = FLT_MAX;
                for (int d = minDisparity; d < maxDisparity; d++) {
                    if (x - d <0 || x - d >cols - 1) {
                        continue;
                    }
                    cost = cuda::std::abs(leftImg.ptr(y)[x] - rightImg.ptr(y)[x - d]);
                    if (cost < minCost) {
                        minCost = cost;
                        k = d;
                    }
                }
                if (minCost > threshod || x - k + 1 > cols - 1 || x - k - 1 < 0) {
                    return;
                }
                float dived = rightImg.ptr(y)[x - k + 1] - rightImg.ptr(y)[x - k - 1];
                if (cuda::std::abs(dived) < 0.001) {
                    dived = 0.001;
                }
                float disparity = k + 2 * (rightImg.ptr(y)[x - k] - leftImg.ptr(y)[x]) / dived;
                if (disparity<minDisparity || disparity>maxDisparity) {
                    return;
                }
                
                Eigen::Vector3f vertex;
                vertex(0, 0) = -1.0f * tx * (x - cx) / (disparity - cxlr);
                vertex(1, 0) = -1.0f * tx * (y - cy) / (disparity - cxlr);
                vertex(2, 0) = -1.0f * tx * f / (disparity - cxlr);
                const Eigen::Vector3f leftVertex = R1_inv * vertex;
                const float depthMapped = leftVertex(2, 0);
                const Eigen::Vector3f mapVec = M1 * leftVertex;
                const int map_x = mapVec(0, 0) / mapVec(2,0);
                const int map_y = mapVec(1, 0) / mapVec(2,0);
              
                if (map_x < cols && map_x >= 0 && map_y >= 0 && map_y < rows) {
                    if (depthMapped < minDepth || depthMapped > maxDepth) {
                        atomicExch(&mapDepth.ptr(map_y)[map_x], 0);
                        //mapDepth.ptr(map_y)[map_x] = 0;
                    }
                    else {
                        atomicExch(&mapDepth.ptr(map_y)[map_x], depthMapped);
                        //mapDepth.ptr(map_y)[map_x] = depthMapped * 5;
                    }
                }
            }
        }

        void depthColorMap(const cv::cuda::GpuMat& leftImg_, const cv::cuda::GpuMat& rightImg_, const int rows, const int cols,
            const int minDisparity, const int maxDisparity, const float minDepth, const float maxDepth, const Eigen::Matrix4f& Q,
            const Eigen::Matrix3f& M3, const Eigen::Matrix3f& R, const Eigen::Vector3f& T, const Eigen::Matrix3f& R1_inv,
            cv::cuda::GpuMat& depthMap, const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            matchAndTriangulateColor_CUDA << <grid, block,0, stream >> > (leftImg_, rightImg_, rows, cols,
                minDisparity, maxDisparity, minDepth, maxDepth, Q,
                M3, R, T, R1_inv, depthMap);
        }

        void depthMap(const cv::cuda::GpuMat& leftImg_, const cv::cuda::GpuMat& rightImg_, const int rows, const int cols,
            const int minDisparity, const int maxDisparity, const float minDepth, const float maxDepth, const Eigen::Matrix4f& Q,
            const Eigen::Matrix3f& M1, const Eigen::Matrix3f& R1_inv, cv::cuda::GpuMat& depthMap, const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            matchAndTriangulate_CUDA << <grid, block, 0, stream >> > (leftImg_, rightImg_, rows, cols,
                minDisparity, maxDisparity, minDepth, maxDepth, Q,
                M1, R1_inv, depthMap);
        }
    }
}

