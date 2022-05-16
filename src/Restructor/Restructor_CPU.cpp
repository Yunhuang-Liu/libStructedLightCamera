#include <Restructor/Restructor_CPU.h>

namespace RestructorType {
    Restructor_CPU::Restructor_CPU(const Info& calibrationInfo_, const int minDisparity_, const int maxDisparity_,
        const float minDepth_, const float maxDepth_, const int threads_) :
        calibrationInfo(calibrationInfo_),
        minDisparity(minDisparity_),
        maxDisparity(maxDisparity_),
        minDepth(minDepth_),
        maxDepth(maxDepth_),
        threads(threads_){
    }

    Restructor_CPU::~Restructor_CPU(){
    }

    void Restructor_CPU::restruction(const cv::Mat& leftAbsImg, const cv::Mat& rightAbsImg, const cv::Mat& colorImg,
        cv::Mat& depthImgOut, cv::Mat& colorImgOut){
        if (depthImgOut.empty()) {
            depthImgOut.create(leftAbsImg.size(),CV_32FC1);
            colorImgOut.create(leftAbsImg.size(),CV_8UC3);
        }
        else {
            depthImgOut.setTo(0);
            colorImgOut.setTo(cv::Vec3b(0, 0, 0));
        }
        getDepthColorMap(leftAbsImg,rightAbsImg,colorImg,depthImgOut,colorImgOut);
    }

    void Restructor_CPU::getDepthColorMap(const cv::Mat& leftAbsImg, const cv::Mat& rightAbsImg, const cv::Mat& colorImg,
        cv::Mat& depthImgOut, cv::Mat& colorImgOut){
        std::vector<std::thread> tasks;
        tasks.resize(threads);
        const int rows = leftAbsImg.rows / threads;
        for(int i=0;i<threads-1;i++){
            tasks[i] = std::thread(&RestructorType::Restructor_CPU::thread_DepthColorMap, this, std::ref(leftAbsImg), std::ref(rightAbsImg), std::ref(colorImg), std::ref(depthImgOut), std::ref(colorImgOut), cv::Point2i(rows * i, rows * (i + 1)));
        }
        tasks[threads-1] = std::thread(&RestructorType::Restructor_CPU::thread_DepthColorMap, this, std::ref(leftAbsImg), std::ref(rightAbsImg), std::ref(colorImg), std::ref(depthImgOut), std::ref(colorImgOut), cv::Point2i(rows * (threads - 1), leftAbsImg.rows));
        for(int i=0;i<threads;i++){
            if(tasks[i].joinable()){
                tasks[i].join();
            }
        }
    }

    void Restructor_CPU::thread_DepthColorMap(const cv::Mat& leftAbsImg, const cv::Mat& righAbstImg,const cv::Mat& colorImg,
        cv::Mat& depthImgOut, cv::Mat& colorImgOut, const cv::Point2i region){
        const cv::Mat& Q = calibrationInfo.Q;
        const float f = Q.at<double>(2,3);
        const float tx = -1.0 / Q.at<double>(3,2);
        const float cxlr = Q.at<double>(3,3) * tx;
        const float cx = -1.0 * Q.at<double>(0,3);
        const float cy = -1.0 * Q.at<double>(1,3);
        const float threshod = 0.1;
        const int rows = leftAbsImg.rows;
        const int cols = leftAbsImg.cols;
        //存放线性回归的实际值
        std::vector<cv::Point2f> linear_return = { cv::Point2f(1,1), cv::Point2f(1, 1), cv::Point2f(1, 1), cv::Point2f(1, 1) };
        //存放回归拟合结果：0：dx，1：dy，2：x，3：y
        std::vector<float> result_Linear = { 1,1,1,1 };
        //线性回归的斜率
        float a;
        //线性回归的截距
        float b;
        //搜寻到的匹配点列号
        int k;
        float minCost;
        float cost;
        float disparity;
        cv::Mat r1Inv = calibrationInfo.R1.inv();
        for (int i = region.x; i < region.y; i++)
        {
            const float* ptr_Left = leftAbsImg.ptr<float>(i);
            const float* ptr_Right = righAbstImg.ptr<float>(i);
            float* ptr_DepthImgOut = depthImgOut.ptr<float>(i);
            cv::Vec3b* ptr_ColorImgOut = colorImgOut.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; j++)
            {
                if(ptr_Left[j] <= 0){
                    ptr_DepthImgOut[j] = 0;
                    ptr_ColorImgOut[j] = cv::Vec3b(0, 0, 0);
                    continue;
                }
                minCost = FLT_MAX;
                for(int d = minDisparity; d < maxDisparity; d++){
                    if(j-d <0 || j-d >cols-1){
                        continue;
                    }
                    cost = std::abs(ptr_Left[j]-ptr_Right[j-d]);
                    if(cost < minCost){
                        k = d;
                        minCost = cost;
                    }
                }
                if(minCost > threshod){
                    ptr_DepthImgOut[j] = 0;
                    ptr_ColorImgOut[j] = cv::Vec3b(0, 0, 0);
                    continue;
                }
                //以左边一位开始连续四点拟合
                linear_return[0] = cv::Point2f(k + 1, ptr_Right[j-k-1]);
                linear_return[1] = cv::Point2f(k, ptr_Right[j-k]);
                linear_return[2] = cv::Point2f(k - 1, ptr_Right[j-k+1]);
                linear_return[3] = cv::Point2f(k - 2, ptr_Right[j-k+2]);
                //线性拟合
                cv::fitLine(linear_return, result_Linear, cv::DIST_L2, 0, 0.01, 0.01);
                a = result_Linear[1] / result_Linear[0];
                b = result_Linear[3] - a * result_Linear[2];
                disparity =(ptr_Left[j] - b) / a;//获取到的亚像素匹配点
                if(disparity<minDisparity || disparity>maxDisparity){
                    ptr_DepthImgOut[j] = 0;
                    ptr_ColorImgOut[j] = cv::Vec3b(0, 0, 0);
                    continue;
                }
                cv::Mat recCameraPoints = (cv::Mat_<double>(3, 1) << -1.0 * tx * (j - cx)/(disparity - cxlr) , -1.0 * tx * (i - cy)/(disparity - cxlr), -1.0 * tx * f/(disparity - cxlr));
                cv::Mat cameraPoints = r1Inv * recCameraPoints;
                const float depth = cameraPoints.at<double>(2, 0);
                if (depth < minDepth || depth > maxDepth) {
                    ptr_DepthImgOut[j] = 0;
                    ptr_ColorImgOut[j] = cv::Vec3b(0, 0, 0);
                    continue;
                }
                cv::Mat result = calibrationInfo.M3 * (calibrationInfo.R * cameraPoints + calibrationInfo.T);
                int x_maped = std::round(result.at<double>(0, 0) / result.at<double>(2, 0));
                int y_maped = std::round(result.at<double>(1, 0) / result.at<double>(2, 0));
                if ((0 < x_maped) && (y_maped < rows) && (0 < y_maped) && (x_maped < cols)) {
                    ptr_DepthImgOut[j] = cameraPoints.at<double>(2,0);
                    ptr_ColorImgOut[j] = colorImg.ptr<cv::Vec3b>(y_maped)[x_maped];
                }
            }
        }
    }
}

