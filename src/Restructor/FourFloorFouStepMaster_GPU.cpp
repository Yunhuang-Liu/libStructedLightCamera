#include <Restructor/FourFloorFouStepMaster_GPU.h>

namespace sl {
    namespace phaseSolver {
        FourFloorFourStepMaster_GPU::FourFloorFourStepMaster_GPU(
                std::vector<cv::Mat> &imgs, const dim3 block_) : block(block_), currentFrame(0) {
            if (imgs.size() > 0) {
                imgs_device.resize(imgs.size());
                for (int i = 0; i < imgs_device.size(); i++) {
                    imgs_device[i].upload(imgs[i]);
                }
            }
            currentFrame++;
        }

        FourFloorFourStepMaster_GPU::~FourFloorFourStepMaster_GPU() {
        }

        void FourFloorFourStepMaster_GPU::getWrapPhaseImg() {
        }

        void FourFloorFourStepMaster_GPU::getUnwrapPhaseImg(
                std::vector<cv::cuda::GpuMat> &unwrapImg, cv::cuda::Stream &pStream) {
            /*
    if(currentFrame == 0)
        PhaseSolverType::cudaFunc::solvePhasePrepare_FourFloorFourStep(imgs_device[1], imgs_device[2], imgs_device[4], imgs_device[5], imgs_device[0], imgs_device[3],centroid_dark, centroid_lightDark, centroid_lightWhite, centroid_white, rows, cols,wrapImg_device, conditionImg_device, floorImg_device, false, block, pStream);
    */
            unwrapImg.resize(1);
            for (int i = 0; i < unwrapImg.size(); i++) {
                unwrapImg[i].create(rows, cols, CV_32FC1);
            }

            if (currentFrame % 2 != 0)
                phaseSolver::cudaFunc::solvePhasePrepare_FourFloorFourStep(imgs_device[4], imgs_device[5], imgs_device[1], imgs_device[2], imgs_device[0], imgs_device[3], threshodVal, threshodAdd, count, rows, cols, wrapImg_device, conditionImg_device, medianFilter_0_, medianFilter_1_, floorImg_0_device, floorImg_1_device, floorImg_device, false, !currentFrame, block, pStream);
            else
                phaseSolver::cudaFunc::solvePhasePrepare_FourFloorFourStep(imgs_device[1], imgs_device[2], imgs_device[4], imgs_device[5], imgs_device[0], imgs_device[3], threshodVal, threshodAdd, count, rows, cols, wrapImg_device, conditionImg_device, medianFilter_0_, medianFilter_1_, floorImg_0_device, floorImg_1_device, floorImg_device, true, !currentFrame, block, pStream);
            phaseSolver::cudaFunc::solvePhase_FourFloorFourStep(floorImg_device, conditionImg_device, wrapImg_device, rows, cols, unwrapImg[0], block, pStream);
            currentFrame++;
        }

        void FourFloorFourStepMaster_GPU::changeSourceImg(std::vector<cv::Mat> &imgs) {
            rows = imgs[0].rows;
            cols = imgs[0].cols;
            imgs_device.resize(imgs.size());
            for (int i = 0; i < imgs_device.size(); i++) {
                imgs_device[i].upload(imgs[i]);
            }
            if (wrapImg_device.empty()) {
                wrapImg_device.create(rows, cols, CV_32FC1);
                floorImg_device.create(rows, cols, CV_8UC1);
                floorImg_0_device.create(rows, cols, CV_32FC1);
                floorImg_1_device.create(rows, cols, CV_32FC1);
                conditionImg_device.create(rows, cols, CV_32FC1);
            }
        }

        void FourFloorFourStepMaster_GPU::changeSourceImg(
                std::vector<cv::Mat> &imgs, cv::cuda::Stream &stream) {
            rows = imgs[0].rows;
            cols = imgs[0].cols;
            if (imgs_device.size() == 0)
                imgs_device.resize(imgs.size());

            if (currentFrame == 0) {
                for (int i = 0; i < imgs_device.size(); i++) {
                    imgs_device[i].upload(imgs[i], stream);
                }
            } else if (currentFrame % 2 != 0) {
                for (int i = 0; i < 3; i++) {
                    imgs_device[i].upload(imgs[i], stream);
                }
            } else {
                for (int i = 3; i < 6; i++) {
                    imgs_device[i].upload(imgs[i], stream);
                }
            }
            if (wrapImg_device.empty()) {
                medianFilter_0_.create(rows, cols, CV_32FC1);
                medianFilter_1_.create(rows, cols, CV_32FC1);
                wrapImg_device.create(rows, cols, CV_32FC1);
                floorImg_device.create(rows, cols, CV_8UC1);
                floorImg_0_device.create(rows, cols, CV_32FC1);
                floorImg_1_device.create(rows, cols, CV_32FC1);
                conditionImg_device.create(rows, cols, CV_32FC1);
            }
        }

        void FourFloorFourStepMaster_GPU::changeSourceImg(
                std::vector<cv::cuda::GpuMat> &imgs) {
            rows = imgs[0].rows;
            cols = imgs[0].cols;

            if (imgs_device.size() == 0)
                imgs_device.resize(imgs.size());

            if (currentFrame == 0) {
                for (int i = 0; i < imgs_device.size(); i++) {
                    imgs_device[i] = imgs[i];
                }
            } else if (currentFrame % 2 != 0) {
                for (int i = 0; i < 3; i++) {
                    imgs_device[i] = imgs[i];
                }
            } else {
                for (int i = 3; i < 6; i++) {
                    imgs_device[i] = imgs[i - 3];
                }
            }

            if (wrapImg_device.empty()) {
                medianFilter_0_.create(rows, cols, CV_32FC1);
                medianFilter_1_.create(rows, cols, CV_32FC1);
                wrapImg_device.create(rows, cols, CV_32FC1);
                floorImg_device.create(rows, cols, CV_8UC1);
                floorImg_0_device.create(rows, cols, CV_32FC1);
                floorImg_1_device.create(rows, cols, CV_32FC1);
                conditionImg_device.create(rows, cols, CV_32FC1);
                count.create(4, 1, CV_32FC1);
                threshodAdd.create(4, 1, CV_32FC1);
                cv::Mat initial = (cv::Mat_<float>(4, 1) << -1.f, -1.f / 3, 1.f / 3, 1.f);
                threshodVal.upload(initial);
            }
        }

        FourFloorFourStepMaster_GPU::FourFloorFourStepMaster_GPU(const dim3 block_) : block(block_), currentFrame(0) {
            /*
    cv::Mat initial = (cv::Mat_<float>(4, 1) << -1.f, -1.f / 3, 1.f / 3, 1.f);
    medianFilter_0_.create(rows, cols, CV_32FC1);
    medianFilter_1_.create(rows, cols, CV_32FC1);
    wrapImg_device.create(rows, cols, CV_32FC1);
    floorImg_device.create(rows, cols, CV_8UC1);
    floorImg_0_device.create(rows, cols, CV_32FC1);
    floorImg_1_device.create(rows, cols, CV_32FC1);
    conditionImg_device.create(rows, cols, CV_32FC1);
    count.create(4, 1, CV_32FC1);
    threshodAdd.create(4, 1, CV_32FC1);
    threshodVal.upload(initial);
    */
        }

        void FourFloorFourStepMaster_GPU::getTextureImg(
                std::vector<cv::cuda::GpuMat> &textureImg) {
            textureImg.clear();
            textureImg.resize(1, cv::cuda::GpuMat(conditionImg_device.size(),
                                                  conditionImg_device.type()));
            textureImg[0] = conditionImg_device;
        }
    }// namespace phaseSolver
}// namespace sl