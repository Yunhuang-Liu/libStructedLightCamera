#include <Restructor/ShiftGrayCodeUnwrapMaster_GPU.h>

namespace PhaseSolverType {
    ShiftGrayCodeUnwrapMaster_GPU::ShiftGrayCodeUnwrapMaster_GPU(std::vector<cv::Mat>& imgs,const cv::Mat& refImgWhite, const dim3 block_): block(block_){
        rows = imgG_1_device.rows;
        cols = imgG_1_device.cols;
        wrapImg1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
        wrapImg2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
        averageImg_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
        conditionImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
        conditionImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
        unwrapImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
        unwrapImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
        floor_K_device = cv::cuda::createContinuous(rows, cols, CV_8UC1);
        img1_1_device.upload(imgs[0]);
        img1_2_device.upload(imgs[1]);
        img1_3_device.upload(imgs[2]);
        img2_1_device.upload(imgs[4]);
        img2_2_device.upload(imgs[5]);
        img2_3_device.upload(imgs[6]);
        imgG_1_device.upload(imgs[3]);
        imgG_2_device.upload(imgs[7]);
        imgG_3_device.upload(imgs[8]);
        imgG_4_device.upload(imgs[9]);
    }

    ShiftGrayCodeUnwrapMaster_GPU::~ShiftGrayCodeUnwrapMaster_GPU() {

    }

    void ShiftGrayCodeUnwrapMaster_GPU::getWrapPhaseImg(cv::cuda::Stream& pStream) {
        PhaseSolverType::cudaFunc::solvePhasePrepare_ShiftGray(img1_1_device, img1_2_device, img1_3_device,
            img2_1_device, img2_2_device, img2_3_device,
            imgG_1_device, imgG_2_device, imgG_3_device, imgG_4_device,
            rows, cols,
            wrapImg1_device, conditionImg_1_device,
            wrapImg2_device, conditionImg_2_device,
            floor_K_device, block, pStream);
    }

    void ShiftGrayCodeUnwrapMaster_GPU::getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& pStream) {
        getWrapPhaseImg(pStream);
        //cudaDeviceSynchronize();
        unwrapImg.resize(2);
        for(int i=0;i<unwrapImg.size();i++){
            unwrapImg[i].create(rows,cols,CV_32FC1);
        }
        PhaseSolverType::cudaFunc::solvePhase_ShiftGray(refImgWhite_device,rows, cols,
            wrapImg1_device, conditionImg_1_device, unwrapImg[0],
            wrapImg2_device, conditionImg_2_device, unwrapImg[1], floor_K_device, block, pStream);
        /*
        cudaDeviceSynchronize();
        std::vector<cv::Mat> test(10);
        wrapImg1_device.download(test[0]);
        wrapImg2_device.download(test[1]);
        conditionImg_1_device.download(test[2]);
        conditionImg_2_device.download(test[3]);
        floor_K_device.download(test[6]);
        unwrapImg_1_.download(test[4]);
        unwrapImg_2_.download(test[5]);
        */
    }

    void ShiftGrayCodeUnwrapMaster_GPU::changeSourceImg(std::vector<cv::Mat>& imgs) {
        if (0 == rows) {
            rows = imgs[0].rows;
            cols = imgs[0].cols;
            wrapImg1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            wrapImg2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            averageImg_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            conditionImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            conditionImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            unwrapImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            unwrapImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            floor_K_device = cv::cuda::createContinuous(rows, cols, CV_8UC1);
        }
        img1_1_device.upload(imgs[0]);
        img1_2_device.upload(imgs[1]);
        img1_3_device.upload(imgs[2]);
        img2_1_device.upload(imgs[4]);
        img2_2_device.upload(imgs[5]);
        img2_3_device.upload(imgs[6]);
        imgG_1_device.upload(imgs[3]);
        imgG_2_device.upload(imgs[7]);
        imgG_3_device.upload(imgs[8]);
        imgG_4_device.upload(imgs[9]);
    }

    void ShiftGrayCodeUnwrapMaster_GPU::changeSourceImg(std::vector<cv::Mat>& imgs, cv::cuda::Stream& cvStream) {
        if (0 == rows) {
            rows = imgs[0].rows;
            cols = imgs[0].cols;
            wrapImg1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            wrapImg2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            averageImg_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            conditionImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            conditionImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            unwrapImg_1_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            unwrapImg_2_device = cv::cuda::createContinuous(rows, cols, CV_32FC1);
            floor_K_device = cv::cuda::createContinuous(rows, cols, CV_8UC1);
        }
        img1_1_device.upload(imgs[0], cvStream);
        img1_2_device.upload(imgs[1], cvStream);
        img1_3_device.upload(imgs[2], cvStream);
        img2_1_device.upload(imgs[4], cvStream);
        img2_2_device.upload(imgs[5], cvStream);
        img2_3_device.upload(imgs[6], cvStream);
        imgG_1_device.upload(imgs[3], cvStream);
        imgG_2_device.upload(imgs[7], cvStream);
        imgG_3_device.upload(imgs[8], cvStream);
        imgG_4_device.upload(imgs[9], cvStream);
    }

    void ShiftGrayCodeUnwrapMaster_GPU::changeSourceImg(std::vector<cv::cuda::GpuMat>& imgs) {
        img1_1_device = imgs[0];
        img1_2_device = imgs[1];
        img1_3_device = imgs[2];
        img2_1_device = imgs[4];
        img2_2_device = imgs[5];
        img2_3_device = imgs[6];
        imgG_1_device = imgs[3];
        imgG_2_device = imgs[7];
        imgG_3_device = imgs[8];
        imgG_4_device = imgs[9];
    }

    ShiftGrayCodeUnwrapMaster_GPU::ShiftGrayCodeUnwrapMaster_GPU(const dim3 block_, const cv::Mat& refImgWhite) : rows(0), cols(0), block(block_), refImgWhite_device(refImgWhite){
        //refImgWhite_device.upload(refImgWhite);
    }

    void ShiftGrayCodeUnwrapMaster_GPU::getTextureImg(std::vector<cv::cuda::GpuMat>& textureImg) {
        textureImg.clear();
        textureImg.resize(1, cv::cuda::GpuMat(averageImg_device.size(), averageImg_device.type()));
        textureImg[0] = averageImg_device;
    }
}
