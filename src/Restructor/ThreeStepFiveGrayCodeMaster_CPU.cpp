#include <Restructor/ThreeStepFiveGrayCodeMaster_CPU.h>

namespace PhaseSolverType {
    ThreeStepFiveGrayCodeMaster_CPU::ThreeStepFiveGrayCodeMaster_CPU(std::vector<cv::Mat>& imgs_, const int threads_) :
        imgs(imgs_),
        threads(threads_) {
    }

    ThreeStepFiveGrayCodeMaster_CPU::~ThreeStepFiveGrayCodeMaster_CPU(){

    }

    void ThreeStepFiveGrayCodeMaster_CPU::getWrapPhaseImg(){
        wrapImg.create(imgs[0].size(),CV_32FC1);
        atan3M(imgs[0],imgs[1],imgs[2],wrapImg,threads);
    }

    void ThreeStepFiveGrayCodeMaster_CPU::caculateAverageImgs(){
        const int rows_ = imgs[0].rows;
        const int cols_ = imgs[0].cols;
        std::vector<std::thread> tasks;
        tasks.resize(threads);
        int rows = rows_ / threads;
        for(int i=0;i<threads-1;i++){
            tasks[i] = std::thread(&ThreeStepFiveGrayCodeMaster_CPU::dataInit_Thread_SIMD,this,rows_,cols_,cv::Point2i(rows*i,rows*(i+1)));
        }
        tasks[threads-1] = std::thread(&ThreeStepFiveGrayCodeMaster_CPU::dataInit_Thread_SIMD,this,rows_,cols_,cv::Point2i(rows*(threads-1),rows_));
        for(int i=0;i<threads;i++){
            if(tasks[i].joinable()){
                tasks[i].join();
            }
        }
    }

    void ThreeStepFiveGrayCodeMaster_CPU::dataInit_Thread_SIMD(const int rows,const int cols,const cv::Point2i region){
        __m256 img_0_data;
        __m256 img_1_data;
        __m256 img_2_data;
        __m256 img_3_data;
        __m256 value_3_data = _mm256_set1_ps(3);
        __m256 average_data;
        __m256 i13_i3_2_data;
        __m256 i22_i1_i3_2_data;
        __m256 condition_data;
        __m256 value_2_data = _mm256_set1_ps(2);
        for(int i=region.x;i<region.y;i++){
            const float* ptr_img_0_ = imgs[0].ptr<float>(i);
            const float* ptr_img_1_ = imgs[1].ptr<float>(i);
            const float* ptr_img_2_ = imgs[2].ptr<float>(i);
            float* ptr_averageImg = averageImg.ptr<float>(i);
            float* ptr_conditionImg = conditionImg.ptr<float>(i);
            for(int j=0;j<cols;j+=8){
                img_0_data = _mm256_load_ps(&ptr_img_0_[j]);
                img_1_data = _mm256_load_ps(&ptr_img_1_[j]);
                img_2_data = _mm256_load_ps(&ptr_img_2_[j]);
                average_data = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(img_0_data,img_1_data),img_2_data),value_3_data);
                i13_i3_2_data = _mm256_mul_ps(_mm256_pow_ps(_mm256_sub_ps(img_0_data,img_2_data), value_2_data), value_3_data);
                i22_i1_i3_2_data = _mm256_pow_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(img_1_data, value_2_data),img_0_data), img_2_data), value_2_data);
                condition_data = _mm256_div_ps(_mm256_sqrt_ps(_mm256_add_ps(i13_i3_2_data, i22_i1_i3_2_data)), value_3_data);
                _mm256_store_ps(&ptr_averageImg[j],average_data);
                _mm256_store_ps(&ptr_conditionImg[j],condition_data);
            }
        }
    }

    void ThreeStepFiveGrayCodeMaster_CPU::mutiplyThreadUnwrap(const int rows,const int cols,const cv::Point2i region,cv::Mat& absolutePhaseImg){
        __m256 leftMove_4_ = _mm256_set1_ps(16);
        __m256 leftMove_3_ = _mm256_set1_ps(8);
        __m256 leftMove_2_ = _mm256_set1_ps(4);
        __m256 leftMove_1_ = _mm256_set1_ps(2);
        __m256 add_1_ = _mm256_set1_ps(1);
        __m256 div_2_ = _mm256_set1_ps(2);
        __m256 compare_Condition_10 =_mm256_set1_ps(5.0);
        __m256 K1;
        __m256 K2;
        __m256 _Counter_PI_Div_2_ = _mm256_set1_ps(-CV_PI/2);
        __m256 _PI_Div_2_ = _mm256_set1_ps(CV_PI/2);
        __m256 _2PI_ = _mm256_set1_ps(CV_2PI);
        __m256 zero = _mm256_set1_ps(0);
        __m256 one = _mm256_set1_ps(1);
        __m256 img_0_Data;
        __m256 img_1_Data;
        __m256 img_2_Data;
        __m256 img_3_Data;
        __m256 img_4_Data;
        __m256 averageData;
        __m256 wrapImgData;
        __m256 conditionData;
        __m256 compareCondition;
        __m256 compareImg_0_;
        __m256 compareImg_1_;
        __m256 compareImg_2_;
        __m256 compareImg_3_;
        __m256 compareImg_4_;
        __m256 condition_CompareData;
        __m256 Img_0_CompareData;
        __m256 Img_1_CompareData;
        __m256 Img_2_CompareData;
        __m256 Img_3_CompareData;
        __m256 Img_4_CompareData;
        __m256 bit4;
        __m256 bit3;
        __m256 bit2;
        __m256 bit1;
        __m256 bit0;
        __m256 lessEqualThan;
        __m256 greaterEqualThan;
        __m256 less_data_greaterThan;
        __m256 data_1_;
        __m256 data_2_;
        __m256 data;
        for (int i = region.x; i < region.y; i++)
        {
            const float* ptr0 = imgs[3].ptr<float>(i);
            const float* ptr1 = imgs[4].ptr<float>(i);
            const float* ptr2 = imgs[5].ptr<float>(i);
            const float* ptr3 = imgs[6].ptr<float>(i);
            const float* ptr4 = imgs[7].ptr<float>(i);
            const float* ptr_Average = averageImg.ptr<float>(i);
            const float* ptr_WrapImg = wrapImg.ptr<float>(i);
            const float* ptr_Condition = conditionImg.ptr<float>(i);
            float* ptr_absoluteImg = absolutePhaseImg.ptr<float>(i);
            for (int j = 0; j < cols; j+=8)
            {
                img_0_Data = _mm256_load_ps(&ptr0[j]);
                img_1_Data = _mm256_load_ps(&ptr1[j]);
                img_2_Data = _mm256_load_ps(&ptr2[j]);
                img_3_Data = _mm256_load_ps(&ptr3[j]);
                img_4_Data = _mm256_load_ps(&ptr4[j]);
                averageData = _mm256_load_ps(&ptr_Average[j]);
                wrapImgData = _mm256_load_ps(&ptr_WrapImg[j]);
                conditionData = _mm256_load_ps(&ptr_Condition[j]);
                compareCondition = _mm256_cmp_ps(conditionData,compare_Condition_10,_CMP_GT_OS);
                compareImg_0_ = _mm256_cmp_ps(img_0_Data,averageData,_CMP_GE_OS);
                compareImg_1_ = _mm256_cmp_ps(img_1_Data,averageData,_CMP_GE_OS);
                compareImg_2_ = _mm256_cmp_ps(img_2_Data,averageData,_CMP_GE_OS);
                compareImg_3_ = _mm256_cmp_ps(img_3_Data,averageData,_CMP_GE_OS);
                compareImg_4_ = _mm256_cmp_ps(img_4_Data,averageData,_CMP_GE_OS);
                Img_0_CompareData = _mm256_and_ps(compareImg_0_,one);
                Img_1_CompareData = _mm256_and_ps(compareImg_1_,one);
                Img_2_CompareData = _mm256_and_ps(compareImg_2_,one);
                Img_3_CompareData = _mm256_and_ps(compareImg_3_,one);
                Img_4_CompareData = _mm256_and_ps(compareImg_4_,one);
                bit4 = _mm256_xor_ps(Img_0_CompareData,zero);
                bit3 = _mm256_xor_ps(Img_1_CompareData,bit4);
                bit2 = _mm256_xor_ps(Img_2_CompareData,bit3);
                bit1 = _mm256_xor_ps(Img_3_CompareData,bit2);
                bit0 = _mm256_xor_ps(Img_4_CompareData,bit1);
                K2 = _mm256_floor_ps(_mm256_div_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(bit4,leftMove_4_),_mm256_mul_ps(bit3,leftMove_3_)),_mm256_mul_ps(bit2,leftMove_2_)),
                                            _mm256_mul_ps(bit1,leftMove_1_)),bit0),add_1_),div_2_));
                bit3 = _mm256_xor_ps(Img_0_CompareData,zero);
                bit2 = _mm256_xor_ps(Img_1_CompareData,bit3);
                bit1 = _mm256_xor_ps(Img_2_CompareData,bit2);
                bit0 = _mm256_xor_ps(Img_3_CompareData,bit1);
                K1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(bit3,leftMove_3_),_mm256_mul_ps(bit2,leftMove_2_)),_mm256_mul_ps(bit1,leftMove_1_)),bit0);
                lessEqualThan = _mm256_and_ps(_mm256_cmp_ps(wrapImgData,_Counter_PI_Div_2_,_CMP_LE_OS),one);
                greaterEqualThan = _mm256_and_ps(_mm256_cmp_ps(wrapImgData,_PI_Div_2_,_CMP_GE_OS),one);
                less_data_greaterThan = _mm256_xor_ps(_mm256_or_ps(lessEqualThan,greaterEqualThan),one);
                data_1_ = _mm256_mul_ps(lessEqualThan,_mm256_fmadd_ps(_2PI_,K2,wrapImgData));
                data_2_ = _mm256_mul_ps(greaterEqualThan,_mm256_fmadd_ps(_2PI_,_mm256_sub_ps(K2,one),wrapImgData));
                data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(less_data_greaterThan,_mm256_fmadd_ps(_2PI_,K1,wrapImgData)),data_1_),data_2_);
                _mm256_store_ps(&ptr_absoluteImg[j],_mm256_mul_ps(data,_mm256_and_ps(compareCondition,one)));
            }
        }
    }

    void ThreeStepFiveGrayCodeMaster_CPU::changeSourceImg(std::vector<cv::Mat>& imgs_){
        imgs = imgs_;
    }

    void ThreeStepFiveGrayCodeMaster_CPU::getUnwrapPhaseImg(cv::Mat& absolutePhaseImg){
        std::vector<std::thread> convertFloatThreads(imgs.size());
        for (int i = 0; i < convertFloatThreads.size(); i++) {
            convertFloatThreads[i] = std::thread([&, i] {
                imgs[i].convertTo(imgs[i], CV_32FC1);
                });
        }
        for (auto& thread : convertFloatThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        absolutePhaseImg.create(imgs[0].size(),CV_32FC1);
        conditionImg.create(imgs[0].size(),CV_32FC1);
        averageImg.create(imgs[0].size(),CV_32FC1);
        getWrapPhaseImg();
        caculateAverageImgs();
        const int rows_ = absolutePhaseImg.rows;
        const int cols_ = absolutePhaseImg.cols;
        std::vector<std::thread> tasks;
        tasks.resize(threads);
        int rows = rows_ / threads;
        for(int i=0;i<threads-1;i++){
            tasks[i] = std::thread(&ThreeStepFiveGrayCodeMaster_CPU::mutiplyThreadUnwrap,this,rows_,cols_,cv::Point2i(rows*i,rows*(i+1)),std::ref(absolutePhaseImg));
        }
        tasks[threads-1] = std::thread(&ThreeStepFiveGrayCodeMaster_CPU::mutiplyThreadUnwrap,this,rows_,cols_,cv::Point2i(rows*(threads-1),rows_),std::ref(absolutePhaseImg));
        for(int i=0;i<threads;i++){
            tasks[i].join();
        }
    }

    ThreeStepFiveGrayCodeMaster_CPU::ThreeStepFiveGrayCodeMaster_CPU(const int threads_) : threads(threads_), imgs(std::vector<cv::Mat>()){

    }

    void ThreeStepFiveGrayCodeMaster_CPU::atan3M(const cv::Mat& firstStep, const cv::Mat& secondStep, const cv::Mat& thirdStep, cv::Mat& wrapImg, const int threads){
        std::vector<std::thread> tasks;
        tasks.resize(threads);
        int rows = firstStep.rows / threads;
        for(int i=0;i<threads-1;i++){
            tasks[i] = std::thread(&ThreeStepFiveGrayCodeMaster_CPU::SIMD_WrapImg,this,std::ref(firstStep),std::ref(secondStep), std::ref(thirdStep), cv::Point2i(rows*i,rows*(i+1)),std::ref(wrapImg));
        }
        tasks[threads-1] = std::thread(&ThreeStepFiveGrayCodeMaster_CPU::SIMD_WrapImg,this, std::ref(firstStep), std::ref(secondStep), std::ref(thirdStep), cv::Point2i(rows*(threads-1), wrapImg.rows),std::ref(wrapImg));
        for(int i=0;i<tasks.size();i++){
            if(tasks[i].joinable()){
                tasks[i].join();
            }
        }
    }

    void ThreeStepFiveGrayCodeMaster_CPU::SIMD_WrapImg(const cv::Mat& firstStep, const cv::Mat& secondStep, const cv::Mat& thirdStep, const cv::Point2i& region,cv::Mat& wrapImg){
        __m256 first_step;
        __m256 second_step;
        __m256 third_step;
        const int cols = firstStep.cols;
        for (size_t i=region.x;i<region.y;i++)
        {
            const float* ptr_first = firstStep.ptr<float>(i);
            const float* ptr_second = secondStep.ptr<float>(i);
            const float* ptr_third = thirdStep.ptr<float>(i);
            float* ptr_wrapImg = wrapImg.ptr<float>(i);
            for (size_t j=0;j<cols;j+=8)
            {
                first_step = _mm256_load_ps(&ptr_first[j]);
                second_step = _mm256_load_ps(&ptr_second[j]);
                third_step = _mm256_load_ps(&ptr_third[j]);
                for(int d=0;d<8;d++){
                    ptr_wrapImg[j+d] = atan2f(std::sqrt(3)* (first_step.m256_f32[d]- third_step.m256_f32[d]),2*second_step.m256_f32[d]-first_step.m256_f32[d]-third_step.m256_f32[d]);
                }
            }
        }
    }

    void ThreeStepFiveGrayCodeMaster_CPU::getWrapPhaseImg(cv::Mat& wrapImg_, cv::Mat& conditionImg_) {
        std::vector<std::thread> convertFloatThreads(imgs.size());
        for (int i = 0; i < convertFloatThreads.size(); i++) {
            convertFloatThreads[i] = std::thread([&, i] {
                imgs[i].convertTo(imgs[i], CV_32FC1);
                });
        }
        for (auto& thread : convertFloatThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        conditionImg.create(imgs[0].size(), CV_32FC1);
        averageImg.create(imgs[0].size(), CV_32FC1);
        getWrapPhaseImg();
        caculateAverageImgs();
        wrapImg_ = wrapImg;
        conditionImg_ = conditionImg;
    }

    void ThreeStepFiveGrayCodeMaster_CPU::getTextureImg(cv::Mat& textureImg) {
        textureImg = averageImg;
    }
}                       