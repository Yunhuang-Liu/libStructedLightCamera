#include <tool/matrixsInfo.h>

#include <cuda/cudaTypeDef.cuh>
#include <cuda_runtime_api.h>

namespace sl {
    namespace tool {
        namespace cudaFunc {
            /**
             * @brief               全图像相位高度映射（CUDA加速优化核函数）
             *
             * @param phase         输入，相位图
             * @param rows          输入，行数
             * @param cols          输入，列数
             * @param intrinsic     输入，内参
             * @param coefficient   输入，八参数
             * @param minDepth      输入，最小深度
             * @param maxDepth      输入，最大深度
             * @param depth         输出，深度图
             */
            __global__ 
            void phaseHeightMapEigCoe_Device(
                const cv::cuda::PtrStep<float> phase, const int rows, const int cols,
                const Eigen::Matrix3f intrinsic, const Eigen::Vector<float,8> coefficient,
                const float minDepth, const float maxDepth, cv::cuda::PtrStep<float> depth) {
                const int x = blockDim.x * blockIdx.x + threadIdx.x;
                const int y = blockDim.y * blockIdx.y + threadIdx.y;
                if (x > cols - 1 || y > rows - 1)
                    return;

                if (phase.ptr(y)[x] == -5.f) {
                    depth.ptr(y)[x] = 0.f;
                    return;
                }

                Eigen::Matrix3f mapL;
                Eigen::Vector3f mapR, cameraPoint;

                mapL(0, 0) = intrinsic(0, 0);
                mapL(0, 1) = 0;
                mapL(0, 2) = intrinsic(0, 2) - x;
                mapL(1, 0) = 0;
                mapL(1, 1) = intrinsic(1, 1);
                mapL(1, 2) = intrinsic(1, 2) - y;
                mapL(2, 0) = coefficient(0, 0) - coefficient(4, 0) * phase.ptr(y)[x];
                mapL(2, 1) = coefficient(1, 0) - coefficient(5, 0) * phase.ptr(y)[x];
                mapL(2, 2) = coefficient(2, 0) - coefficient(6, 0) * phase.ptr(y)[x];

                mapR(0, 0) = 0;
                mapR(1, 0) = 0;
                mapR(2, 0) = coefficient(7, 0) * phase.ptr(y)[x] - coefficient(3, 0);

                cameraPoint = mapL.inverse() * mapR;
                depth.ptr(y)[x] = cameraPoint.z();    
            }

            /**
             * @brief                   反向投影过滤及精度细化（CUDA加速优化核函数）
             *
             * @param phase             输入，相位图
             * @param depth             输入，深度图
             * @param firstWrap         输入，第一个辅助相机包裹相位图
             * @param firstCondition    输入，第一个辅助相机调制图
             * @param secondWrap        输入，第二个辅助相机包裹相位图
             * @param secondConditon    输入，第二个辅助相机调制图
             * @param rows              输入，行数
             * @param cols              输入，列数
             * @param intrinsicInvD     输入，深度相机内参矩阵逆矩阵
             * @param intrinsicF        输入，第一个辅助相机内参
             * @param intrinsicS        输入，第二个辅助相机内参
             * @param RDtoFirst         输入，深度相机到第一个辅助相机旋转矩阵
             * @param TDtoFirst         输入，深度相机到第一个辅助相机平移矩阵
             * @param RDtoSecond        输入，深度相机到第二个辅助相机旋转矩阵
             * @param TDtoSecond        输入，深度相机到第二个辅助相机平移矩阵
             * @param PL                输入，深度相机投影矩阵
             * @param PR                输入，第一个辅助相机投影矩阵
             * @param threshod          输入，去除背景用阈值（一般为2到3个相邻相位差值）
             * @param epilineA          输入，深度相机图像点在第一个辅助相机图片下的极线系数A
             * @param epilineB          输入，深度相机图像点在第一个辅助相机图片下的极线系数B
             * @param epilineC          输入，深度相机图像点在第一个辅助相机图片下的极线系数C
             * @param depthRefine       输出，细化的深度图
             */
            __global__ 
            void reverseMappingRefine_Device(const cv::cuda::PtrStep<float> phase, const cv::cuda::PtrStep<float> depth,
                const cv::cuda::PtrStep<float> firstWrap, const cv::cuda::PtrStep<float> firstCondition,
                const cv::cuda::PtrStep<float> secondWrap, const cv::cuda::PtrStep<float> secondCondition,
                const int rows, const int cols,
                const Eigen::Matrix3f intrinsicInvD, const Eigen::Matrix3f intrinsicF, const Eigen::Matrix3f intrinsicS,
                const Eigen::Matrix3f RDtoFirst, const Eigen::Vector3f TDtoFirst,
                const Eigen::Matrix3f RDtoSecond, const Eigen::Vector3f TDtoSecond,
                const Eigen::Matrix4f PL, const Eigen::Matrix4f PR, const float threshod, 
                const cv::cuda::PtrStep<float> epilineA, const cv::cuda::PtrStep<float> epilineB, const cv::cuda::PtrStep<float> epilineC,
                cv::cuda::PtrStep<float> depthRefine) {
                const int x = blockDim.x * blockIdx.x + threadIdx.x;
                const int y = blockDim.y * blockIdx.y + threadIdx.y;
                
                if (x > cols - 1 || y > rows - 1)
                    return;

                if (depth.ptr(y)[x] == 0.f) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                }

                Eigen::Vector3f imgPoint(x, y ,1);
                Eigen::Vector3f cameraPointNormalized = intrinsicInvD * imgPoint * depth.ptr(y)[x];
                Eigen::Vector3f secondCameraPoint = RDtoFirst * cameraPointNormalized + TDtoFirst;
                Eigen::Vector3f thirdCameraPoint = RDtoSecond * cameraPointNormalized + TDtoSecond;
                Eigen::Vector3f secondImgPoint = intrinsicF * secondCameraPoint;
                Eigen::Vector3f thirdImgPoint = intrinsicS * thirdCameraPoint;
                const int locXF = secondImgPoint(0, 0) / secondImgPoint(2, 0);
                const int locYF = secondImgPoint(1, 0) / secondImgPoint(2, 0);
                const int locXS = thirdImgPoint(0, 0) / thirdImgPoint(2, 0);
                const int locYS = thirdImgPoint(1, 0) / thirdImgPoint(2, 0);

                if (locXF < 0 || locXF > cols - 1 || locYF < 0 || locYF > rows  - 1||
                    locXS < 0 || locXS > cols - 1 || locYS < 0 || locYS  > rows - 1) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                }

                if (firstCondition.ptr(locYF)[locXF] < 10.f || secondCondition.ptr(locYS)[locXS] < 10.f) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                }

                const int floorK = cuda::std::floorf(phase.ptr(y)[x] / CV_2PI);
                const float disparityDF = cuda::std::abs(firstWrap.ptr(locYF)[locXF] + floorK * CV_2PI + CV_PI - phase.ptr(y)[x]);
                const float disparityDS = cuda::std::abs(secondWrap.ptr(locYS)[locXS] + floorK * CV_2PI + CV_PI - phase.ptr(y)[x]);

                if (!((disparityDF < threshod && disparityDS < threshod) ||
                      (disparityDF < threshod && cuda::std::abs(disparityDS - CV_2PI) < threshod) ||
                      (disparityDS < threshod && cuda::std::abs(disparityDF - CV_2PI) < threshod) ||
                      (cuda::std::abs(disparityDF - CV_2PI) < threshod && cuda::std::abs(disparityDS - CV_2PI) < threshod))) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                } 
                
                //找出最有可能的点,从左到右依次为：置信度，X坐标
                float minDiffWrap = FLT_MAX;
                float locXSubpixel = locXF;
                float locYSubpixel = locYF;
                float phaseCoarse = firstWrap.ptr(locYF)[locXF] + floorK * CV_2PI + CV_PI;
                const float believeThreshod = 1.5f;
                for (int i = -5; i < 6; ++i) {
                    const int indexY = locYF + i;
                    if (indexY > rows - 1)
                        continue;
                    for (int j = -5; j < 6; ++j) {
                        const int indexX = locXF + j;
                        if (indexX > cols - 1)
                            continue;
                        if (firstCondition.ptr(indexY)[indexX] < 10.f)
                            continue;
                        const float believeVal = cuda::std::abs(epilineA.ptr(y)[x] * indexX + epilineB.ptr(y)[x] * indexY + epilineC.ptr(y)[x]);

                        if (believeVal < believeThreshod) {
                            float phaseVal = firstWrap.ptr(indexY)[indexX] + floorK * CV_2PI + CV_PI;

                            if (phaseVal - phase.ptr(y)[x] > CV_PI)
                                phaseVal = phaseVal - CV_2PI;
                            else if (phaseVal - phase.ptr(y)[x] < -CV_PI)
                                phaseVal = phaseVal + CV_2PI;

                            const float diffWrap = cuda::std::abs(phaseVal - phase.ptr(y)[x]);

                            if (diffWrap < minDiffWrap) {
                                minDiffWrap = diffWrap;
                                locXSubpixel = indexX;
                                locYSubpixel = indexY;
                                phaseCoarse = phaseVal;
                            }
                        }
                    }
                }

                if (locXSubpixel - 1 >= 0 && locXSubpixel + 1 < cols) {              
                    //线性拟合
                    if (phaseCoarse > phase.ptr(y)[x]) {
                        float phaseValLhs = firstWrap.ptr(locYSubpixel)[(int) locXSubpixel - 1] + floorK * CV_2PI + CV_PI;

                        if (phaseValLhs - phase.ptr(y)[x] > CV_PI)
                            phaseValLhs = phaseValLhs - CV_2PI;
                        else if (phaseValLhs - phase.ptr(y)[x] < -CV_PI)
                            phaseValLhs = phaseValLhs + CV_2PI;

                        locXSubpixel = locXSubpixel - 1 + (phase.ptr(y)[x] - phaseValLhs) / (phaseCoarse - phaseValLhs);
                    } 
                    else {
                        float phaseValRhs = firstWrap.ptr(locYSubpixel)[(int) locXSubpixel + 1] + floorK * CV_2PI + CV_PI;

                        if (phaseValRhs - phase.ptr(y)[x] > CV_PI)
                            phaseValRhs = phaseValRhs - CV_2PI;
                        else if (phaseValRhs - phase.ptr(y)[x] < -CV_PI)
                            phaseValRhs = phaseValRhs + CV_2PI;

                        locXSubpixel = locXSubpixel + 1 - (phaseValRhs - phase.ptr(y)[x]) / (phaseValRhs - phaseCoarse);
                    }
                }

                Eigen::Matrix3f LC;
                Eigen::Vector3f RC;
                LC << PL(0, 0) - x * PL(2, 0), PL(0, 1) - x * PL(2, 1), PL(0, 2) - x * PL(2, 2),
                      PL(1, 0) - y * PL(2, 0), PL(1, 1) - y * PL(2, 1), PL(1, 2) - y * PL(2, 2),
                        PR(0, 0) - locXSubpixel * PR(2, 0), PR(0, 1) - locXSubpixel * PR(2, 1), PR(0, 2) - locXSubpixel * PR(2, 2);
                RC << x * PL(2, 3) - PL(0, 3),
                      y * PL(2, 3) - PL(1, 3),
                        locXSubpixel * PR(2, 3) - PR(0, 3);
                Eigen::Vector3f result = LC.inverse() * RC;

                depthRefine.ptr(y)[x] = result(2, 0);
            }

            /**
             * @brief                   反向投影过滤及精度细化（CUDA加速优化核函数）
             *
             * @param phase             输入，相位图
             * @param depth             输入，深度图
             * @param firstWrap         输入，第一个辅助相机包裹相位图
             * @param firstCondition    输入，第一个辅助相机调制图
             * @param secondWrap        输入，第二个辅助相机包裹相位图
             * @param secondConditon    输入，第二个辅助相机调制图
             * @param rows              输入，行数
             * @param cols              输入，列数
             * @param intrinsicInvD     输入，深度相机内参矩阵逆矩阵
             * @param intrinsicF        输入，第一个辅助相机内参
             * @param intrinsicS        输入，第二个辅助相机内参
             * @param RDtoFirst         输入，深度相机到第一个辅助相机旋转矩阵
             * @param TDtoFirst         输入，深度相机到第一个辅助相机平移矩阵
             * @param RDtoSecond        输入，深度相机到第二个辅助相机旋转矩阵
             * @param TDtoSecond        输入，深度相机到第二个辅助相机平移矩阵
             * @param PL                输入，深度相机投影矩阵
             * @param PR                输入，第一个辅助相机投影矩阵
             * @param threshod          输入，去除背景用阈值（一般为2到3个相邻相位差值）
             * @param epilineA          输入，深度相机图像点在第一个辅助相机图片下的极线系数A
             * @param epilineB          输入，深度相机图像点在第一个辅助相机图片下的极线系数B
             * @param epilineC          输入，深度相机图像点在第一个辅助相机图片下的极线系数C
             * @param depthRefine       输出，细化的深度图
             */
            __global__ void reverseMappingRefineTest_Device(const cv::cuda::PtrStep<float> phase, const cv::cuda::PtrStep<float> depth,
                                                            const cv::cuda::PtrStep<float> firstWrap, const cv::cuda::PtrStep<float> firstCondition,
                                                            const int rows, const int cols,
                                                            const Eigen::Matrix3f intrinsicInvD, const Eigen::Matrix3f intrinsicF,
                                                            const Eigen::Matrix3f RDtoFirst, const Eigen::Vector3f TDtoFirst,
                                                            const Eigen::Matrix4f PL, const Eigen::Matrix4f PR, const float threshod,
                                                            const cv::cuda::PtrStep<float> epilineA, const cv::cuda::PtrStep<float> epilineB, const cv::cuda::PtrStep<float> epilineC,
                                                            cv::cuda::PtrStep<float> depthRefine) {
                const int x = blockDim.x * blockIdx.x + threadIdx.x;
                const int y = blockDim.y * blockIdx.y + threadIdx.y;

                if (x > cols - 1 || y > rows - 1)
                    return;

                if (depth.ptr(y)[x] == 0.f) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                }

                Eigen::Vector3f imgPoint(x, y, 1);
                Eigen::Vector3f cameraPointNormalized = intrinsicInvD * imgPoint * depth.ptr(y)[x];
                Eigen::Vector3f secondCameraPoint = RDtoFirst * cameraPointNormalized + TDtoFirst;
                Eigen::Vector3f secondImgPoint = intrinsicF * secondCameraPoint;
                const int locXF = secondImgPoint(0, 0) / secondImgPoint(2, 0);
                const int locYF = secondImgPoint(1, 0) / secondImgPoint(2, 0);

                if (locXF < 0 || locXF > cols - 1 || locYF < 0 || locYF > rows - 1) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                }
                /*
                if (x == 950 && y == 710) {
                    printf("the loc: %d, %d \n", locXF, locYF);
                }

                if (x == 950 && y == 710) {
                    printf("the distance: %f \n", cuda::std::abs(epilineA.ptr(y)[x] * locXF + epilineB.ptr(y)[x] * locYF + epilineC.ptr(y)[x]));
                }
                */
                printf("the distance: %f \n", cuda::std::abs(epilineA.ptr(y)[x] * locXF + epilineB.ptr(y)[x] * locYF + epilineC.ptr(y)[x]));
                //调制约束+极线约束
                if (firstCondition.ptr(locYF)[locXF] < 10.f || cuda::std::abs(epilineA.ptr(y)[x] * locXF + epilineB.ptr(y)[x] * locYF + epilineC.ptr(y)[x]) > 5.f) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                }

                const int floorK = cuda::std::floorf(phase.ptr(y)[x] / CV_2PI);
                const float disparityDF = cuda::std::abs(firstWrap.ptr(locYF)[locXF] + floorK * CV_2PI + CV_PI - phase.ptr(y)[x]);

                if (!(disparityDF < threshod ||
                      cuda::std::abs(disparityDF - CV_2PI) < threshod)) {
                    depthRefine.ptr(y)[x] = 0.f;
                    return;
                }

                //找出最有可能的点,从左到右依次为：置信度，X坐标
                float minDiffWrap = FLT_MAX;
                float locXSubpixel = locXF;
                float locYSubpixel = locYF;
                int stepLocX = locXF;
                int stepLocY = locYF;

                float phaseVal = firstWrap.ptr(stepLocY)[stepLocX] + floorK * CV_2PI + CV_PI;
                if (phaseVal - phase.ptr(y)[x] > CV_PI)
                    phaseVal = phaseVal - CV_2PI;
                else if (phaseVal - phase.ptr(y)[x] < -CV_PI)
                    phaseVal = phaseVal + CV_2PI;

                float phaseCost, stepPhaseCost = FLT_MAX;
                const int directionX = phaseVal < phase.ptr(y)[x] ? 1 : -1;
                while (true) {
                    phaseVal = firstWrap.ptr(stepLocY)[stepLocX] + floorK * CV_2PI + CV_PI;
                    if (phaseVal - phase.ptr(y)[x] > CV_PI)
                        phaseVal = phaseVal - CV_2PI;
                    else if (phaseVal - phase.ptr(y)[x] < -CV_PI)
                        phaseVal = phaseVal + CV_2PI;

                    phaseCost = cuda::std::abs(phaseVal - phase.ptr(y)[x]);

                    if (phaseCost > stepPhaseCost) {
                        break;
                    }
                    else {
                        stepPhaseCost = phaseCost;
                        locXSubpixel = stepLocX;
                    }

                    stepLocX += directionX;
                }

                float stepEpilineCost = FLT_MAX;
                float epilineCostVal = epilineA.ptr(y)[x] * locXSubpixel + epilineB.ptr(y)[x] * stepLocY + epilineC.ptr(y)[x];
                const int directionY = epilineCostVal > 0 ? -1 : 1;
                while (true) {
                    epilineCostVal = epilineA.ptr(y)[x] * locXSubpixel + epilineB.ptr(y)[x] * stepLocY + epilineC.ptr(y)[x];

                    if (cuda::std::abs(epilineCostVal) > stepEpilineCost) {
                        break;
                    }
                    else {
                        stepEpilineCost = cuda::std::abs(epilineCostVal);
                        locYSubpixel = stepLocY;
                    }

                    stepLocY += directionY;
                }

                float phaseCoarse = firstWrap.ptr((int)locYSubpixel)[(int)locXSubpixel] + floorK * CV_2PI + CV_PI;
                if (phaseCoarse - phase.ptr(y)[x] > CV_PI)
                    phaseCoarse -= CV_2PI;
                else if (phaseCoarse - phase.ptr(y)[x] < -CV_PI)
                    phaseCoarse += CV_2PI;

                if (locXSubpixel - 1 >= 0 && locXSubpixel + 1 < cols) {
                    //线性拟合
                    if (phaseCoarse > phase.ptr(y)[x]) {
                        float phaseValLhs = firstWrap.ptr((int) locYSubpixel)[(int) locXSubpixel - 1] + floorK * CV_2PI + CV_PI;

                        if (phaseValLhs - phase.ptr(y)[x] > CV_PI)
                            phaseValLhs = phaseValLhs - CV_2PI;
                        else if (phaseValLhs - phase.ptr(y)[x] < -CV_PI)
                            phaseValLhs = phaseValLhs + CV_2PI;

                        locXSubpixel = locXSubpixel - 1 + (phase.ptr(y)[x] - phaseValLhs) / (phaseCoarse - phaseValLhs);
                    } else {
                        float phaseValRhs = firstWrap.ptr(locYSubpixel)[(int) locXSubpixel + 1] + floorK * CV_2PI + CV_PI;

                        if (phaseValRhs - phase.ptr(y)[x] > CV_PI)
                            phaseValRhs = phaseValRhs - CV_2PI;
                        else if (phaseValRhs - phase.ptr(y)[x] < -CV_PI)
                            phaseValRhs = phaseValRhs + CV_2PI;

                        locXSubpixel = locXSubpixel + 1 - (phaseValRhs - phase.ptr(y)[x]) / (phaseValRhs - phaseCoarse);
                    }
                }

                Eigen::Matrix3f LC;
                Eigen::Vector3f RC;
                LC << PL(0, 0) - x * PL(2, 0), PL(0, 1) - x * PL(2, 1), PL(0, 2) - x * PL(2, 2),
                        PL(1, 0) - y * PL(2, 0), PL(1, 1) - y * PL(2, 1), PL(1, 2) - y * PL(2, 2),
                        PR(0, 0) - locXSubpixel * PR(2, 0), PR(0, 1) - locXSubpixel * PR(2, 1), PR(0, 2) - locXSubpixel * PR(2, 2);
                RC << x * PL(2, 3) - PL(0, 3),
                        y * PL(2, 3) - PL(1, 3),
                        locXSubpixel * PR(2, 3) - PR(0, 3);
                Eigen::Vector3f result = LC.inverse() * RC;

                depthRefine.ptr(y)[x] = result(2, 0);
            }

            /**
             * @brief               全图像相位高度映射（CUDA加速优化核函数）
             *
             * @param depth         输入，深度图
             * @param textureSrc    输入，纹理相机采集的纹理图
             * @param rows          输入，行数
             * @param cols          输入，列数
             * @param intrinsicInvD 输入，深度相机内参矩阵逆矩阵
             * @param intrinsicT    输入，纹理相机内参
             * @param rotateDToT    输入，深度相机到纹理相机的旋转矩阵
             * @param translateDtoT 输入，深度相机到纹理相机的平移矩阵
             * @param textureMapped 输出，映射到深度相机下的纹理
             */
            __global__
            void reverseMappingTexture_Device(const cv::cuda::PtrStep<float> depth, const cv::cuda::PtrStep<uchar3> textureSrc,
                const int rows, const int cols,
                const Eigen::Matrix3f intrinsicInvD, const Eigen::Matrix3f intrinsicT,
                const Eigen::Matrix3f rotateDToT, const Eigen::Vector3f translateDtoT,
                cv::cuda::PtrStep<uchar3> textureMapped) {
                const int x = blockDim.x * blockIdx.x + threadIdx.x;
                const int y = blockDim.y * blockIdx.y + threadIdx.y;

                if (x > cols - 1 || y > rows - 1)
                    return;

                if (depth.ptr(y)[x] == 0.f)
                    return;

                Eigen::Vector3f imgPoint(x * depth.ptr(y)[x], y * depth.ptr(y)[x], depth.ptr(y)[x]);
                Eigen::Vector3f texturePoint = intrinsicT * (rotateDToT * (intrinsicInvD * imgPoint) + translateDtoT);

                const int xTexture = texturePoint(0, 0) / texturePoint(2, 0);
                const int yTexture = texturePoint(1, 0) / texturePoint(2, 0);

                if (xTexture < 0 || xTexture > cols - 1 || yTexture < 0 || yTexture > rows - 1)
                    return;
                
                textureMapped.ptr(y)[x] = textureSrc.ptr(yTexture)[xTexture];
            }

            /**
             * @brief           从四步相移图像计算纹理图像
             * 
             * @param img_0_    输入，相移图像1
             * @param img_1_    输入，相移图像2
             * @param img_2_    输入，相移图像3
             * @param img_3_    输入，相移图像4
             * @param rows      输入，行数
             * @param cols      输入，列数
             * @param texture   输出，纹理图像
             */
            __global__
            void averageTextureFour_Device(const cv::cuda::PtrStep<uchar3> img_0_, const cv::cuda::PtrStep<uchar3> img_1_,
                const cv::cuda::PtrStep<uchar3> img_2_, const cv::cuda::PtrStep<uchar3> img_3_,
                const int rows, const int cols, cv::cuda::PtrStep<uchar3> texture) {
                const int x = blockDim.x * blockIdx.x + threadIdx.x;
                const int y = blockDim.y * blockIdx.y + threadIdx.y;

                if (x > cols - 1 || y > rows - 1)
                    return;

                texture.ptr(y)[x].x = (img_0_.ptr(y)[x].x + img_1_.ptr(y)[x].x + img_2_.ptr(y)[x].x + img_3_.ptr(y)[x].x) / 4;
                texture.ptr(y)[x].y = (img_0_.ptr(y)[x].y + img_1_.ptr(y)[x].y + img_2_.ptr(y)[x].y + img_3_.ptr(y)[x].y) / 4;
                texture.ptr(y)[x].z = (img_0_.ptr(y)[x].z + img_1_.ptr(y)[x].z + img_2_.ptr(y)[x].z + img_3_.ptr(y)[x].z) / 4;
            }

            /**
             * @brief           从三步相移图像计算纹理图像
             * 
             * @param img_0_    输入，相移图像1
             * @param img_1_    输入，相移图像2
             * @param img_2_    输入，相移图像3
             * @param rows      输入，行数
             * @param cols      输入，列数
             * @param texture   输出，纹理图像
             */
            __global__ void averageTextureThree_Device(const cv::cuda::PtrStep<uchar3> img_0_, const cv::cuda::PtrStep<uchar3> img_1_,
                const cv::cuda::PtrStep<uchar3> img_2_, const int rows, const int cols, 
                cv::cuda::PtrStep<uchar3> texture) {
                const int x = blockDim.x * blockIdx.x + threadIdx.x;
                const int y = blockDim.y * blockIdx.y + threadIdx.y;

                if (x > cols - 1 || y > rows - 1)
                    return;

                texture.ptr(y)[x].x = (img_0_.ptr(y)[x].x + img_1_.ptr(y)[x].x + img_2_.ptr(y)[x].x) / 3;
                texture.ptr(y)[x].y = (img_0_.ptr(y)[x].y + img_1_.ptr(y)[x].y + img_2_.ptr(y)[x].y) / 3;
                texture.ptr(y)[x].z = (img_0_.ptr(y)[x].z + img_1_.ptr(y)[x].z + img_2_.ptr(y)[x].z) / 3;
            }

            void phaseHeightMapEigCoe(const cv::cuda::GpuMat &phase, 
                const Eigen::Matrix3f &intrinsic, const Eigen::Vector<float,8> &coefficient,
                const float minDepth, const float maxDepth,
                cv::cuda::GpuMat &depth, 
                const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null()) {
                
                CV_Assert(!phase.empty() && phase.type() == CV_32FC1);

                if (depth.empty())
                    depth.create(phase.rows, phase.cols, CV_32FC1);
                depth.setTo(0.f);

                const int rows = phase.rows;
                const int cols = phase.cols;

                const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                phaseHeightMapEigCoe_Device<<<grid, block, 0, stream>>>(
                    phase, phase.rows, phase.cols,
                    intrinsic, coefficient,
                    minDepth, maxDepth, depth);
            }

            void reverseMappingRefine(const cv::cuda::GpuMat &phase, const cv::cuda::GpuMat &depth,
                const cv::cuda::GpuMat &firstWrap, const cv::cuda::GpuMat &firstCondition,
                const cv::cuda::GpuMat &secondWrap, const cv::cuda::GpuMat &secondCondition,
                const Eigen::Matrix3f &intrinsicInvD, const Eigen::Matrix3f &intrinsicF, const Eigen::Matrix3f &intrinsicS,
                const Eigen::Matrix3f &RDtoFirst, const Eigen::Vector3f &TDtoFirst,
                const Eigen::Matrix3f &RDtoSecond, const Eigen::Vector3f &TDtoSecond,
                const Eigen::Matrix4f &PL, const Eigen::Matrix4f &PR, const float threshod, 
                const cv::cuda::GpuMat &epilineA, const cv::cuda::GpuMat &epilineB, const cv::cuda::GpuMat &epilineC,
                cv::cuda::GpuMat &depthRefine,
                const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null()) {                
                CV_Assert(!phase.empty() && phase.type() == CV_32FC1 &&
                          !depth.empty() && depth.type() == CV_32FC1);

                if (depthRefine.empty())
                    depthRefine.create(phase.rows, phase.cols, CV_32FC1);
                depthRefine.setTo(0.f);

                const int rows = phase.rows;
                const int cols = phase.cols;

                const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                auto stream = cv::cuda::StreamAccessor::getStream(cvStream);
                reverseMappingRefine_Device<<<grid, block, 0, stream>>>(phase, depth,
                                                                 firstWrap, firstCondition,
                                                                 secondWrap, secondCondition,
                                                                 rows, cols,
                                                                 intrinsicInvD, intrinsicF, intrinsicS,
                                                                 RDtoFirst, TDtoFirst,
                                                                 RDtoSecond, TDtoSecond,
                                                                 PL, PR, threshod, 
                                                                 epilineA, epilineB, epilineC,
                                                                 depthRefine);
            }

            void reverseMappingRefineNew(const cv::cuda::GpuMat &phase, const cv::cuda::GpuMat &depth,
                const cv::cuda::GpuMat &wrap, const cv::cuda::GpuMat &condition,
                const Eigen::Matrix3f &intrinsicInvD, const Eigen::Matrix3f &intrinsicR,
                const Eigen::Matrix3f &RDtoR, const Eigen::Vector3f &TDtoR,
                const Eigen::Matrix4f &PL, const Eigen::Matrix4f &PR, const float threshod,
                const cv::cuda::GpuMat &epilineA, const cv::cuda::GpuMat &epilineB, const cv::cuda::GpuMat &epilineC,
                cv::cuda::GpuMat &depthRefine,
                const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null()) {
                CV_Assert(!phase.empty() && phase.type() == CV_32FC1 &&
                          !depth.empty() && depth.type() == CV_32FC1);

                if (depthRefine.empty())
                    depthRefine.create(phase.rows, phase.cols, CV_32FC1);
                depthRefine.setTo(0.f);

                const int rows = phase.rows;
                const int cols = phase.cols;

                const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                auto stream = cv::cuda::StreamAccessor::getStream(cvStream);
                reverseMappingRefineTest_Device<<<grid, block, 0, stream>>>(phase, depth,
                                                                        wrap, condition,
                                                                        rows, cols,
                                                                        intrinsicInvD, intrinsicR, 
                                                                        RDtoR, TDtoR,
                                                                        PL, PR, threshod,
                                                                        epilineA, epilineB, epilineC,
                                                                        depthRefine);
            }

            void reverseMappingTexture(const cv::cuda::GpuMat& depth, const cv::cuda::GpuMat& textureSrc,
                const Eigen::Matrix3f& intrinsicInvD, const Eigen::Matrix3f& intrinsicT,
                const Eigen::Matrix3f& rotateDToT, const Eigen::Vector3f& translateDtoT,
                cv::cuda::GpuMat& textureMapped,
                const dim3 block = dim3(32, 8), cv::cuda::Stream& cvStream = cv::cuda::Stream::Null()) {
                CV_Assert(depth.type() == CV_32FC1 && !depth.empty() && 
                          textureSrc.type() == CV_8UC3 && !textureSrc.empty());

                if (textureMapped.empty())
                    textureMapped.create(textureSrc.size(), CV_8UC3);
                textureMapped.setTo(cv::Scalar(0.f, 0.f, 0.f));

                const int rows = textureSrc.rows;
                const int cols = textureSrc.cols;
                const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                reverseMappingTexture_Device<<<grid, block, 0, stream>>>(depth, textureSrc,
                                                                         rows, cols,
                                                                         intrinsicInvD, intrinsicT,
                                                                         rotateDToT, translateDtoT,
                                                                         textureMapped);
            }

            void averageTexture(std::vector<cv::cuda::GpuMat>& imgs, cv::cuda::GpuMat& texture,
                const dim3 block = dim3(32, 8), cv::cuda::Stream &cvStream = cv::cuda::Stream::Null()) {
                CV_Assert(imgs.size() == 3 || imgs.size() == 4);

                if (texture.empty())
                    texture.create(imgs[0].size(), CV_8UC3);
                texture.setTo(cv::Scalar(0, 0, 0));

                const int rows = texture.rows;
                const int cols = texture.cols;
                const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                if (imgs.size() == 3)
                    averageTextureThree_Device<<<grid, block, 0, stream>>>(imgs[0], imgs[1], imgs[2], rows, cols, texture);
                if (imgs.size() == 4)
                    averageTextureFour_Device<<<grid, block, 0, stream>>>(imgs[0], imgs[1], imgs[2], imgs[3], rows, cols, texture);
            }

        }// namespace cudaFunc
    }// namespace tool
}// namespace sl