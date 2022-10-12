#include <phaseSolver/fourStepRefPlainMaster_GPU.h>

using namespace sl::wrapCreator;

namespace sl {
    namespace phaseSolver {
        FourStepRefPlainMaster_GPU::FourStepRefPlainMaster_GPU(const cv::Mat refPlain, const bool isFarestMode, 
            const dim3 block_) : m_block(block_), m_refPlainImg(refPlain), wrapCreator(new WrapCreator_GPU),
            m_isCounter(false), m_isFarestPlain(isFarestMode){
        }

        FourStepRefPlainMaster_GPU::~FourStepRefPlainMaster_GPU() {
        }

        void FourStepRefPlainMaster_GPU::getUnwrapPhaseImg(
            std::vector<cv::cuda::GpuMat> &unwrapImg, cv::cuda::Stream &pStream) {
            unwrapImg.resize(1, cv::cuda::GpuMat(m_refPlainImg.rows, m_refPlainImg.cols, CV_32FC1, cv::Scalar(-5.f)));
            sl::wrapCreator::WrapCreator::WrapParameter parameter(m_block);
            wrapCreator->getWrapImg(m_imgs, m_wrapImg, m_conditionImg, m_isCounter, pStream, parameter);

            phaseSolver::cudaFunc::refPlainSolvePhase(m_wrapImg, m_conditionImg, 
                m_refPlainImg, m_refPlainImg.rows, m_refPlainImg.cols,
                unwrapImg[0],m_isFarestPlain,
                m_block,pStream);
        }

        void FourStepRefPlainMaster_GPU::changeSourceImg(
            std::vector<cv::cuda::GpuMat> &imgs) {
            m_imgs.resize(imgs.size());
            for (int i = 0; i < m_imgs.size(); ++i) {
                m_imgs[i] = imgs[i];
            }
        }

        void FourStepRefPlainMaster_GPU::getTextureImg(
            std::vector<cv::cuda::GpuMat> &textureImg) {
            textureImg.clear();
            textureImg.resize(1);
            //TODO(@1369215984):若无法从浮点型转成三通道彩色，考虑再多做一次转换
            //m_conditionImg.convertTo(textureImg[0], CV_8UC1);
            cv::cuda::cvtColor(m_conditionImg, textureImg[0], cv::COLOR_GRAY2BGR);
        }
    }// namespace phaseSolver
}// namespace sl
