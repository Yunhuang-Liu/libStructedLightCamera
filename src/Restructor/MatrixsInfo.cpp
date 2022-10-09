#include <Restructor/MatrixsInfo.h>

namespace SL {
    MatrixsInfo::MatrixsInfo(std::string intrinsicDir, std::string extrinsicDir) {
        cv::FileStorage readYml(intrinsicDir, cv::FileStorage::READ);
        readYml["M1"] >> myInfo.M1;
        readYml["D1"] >> myInfo.D1;
        readYml["M2"] >> myInfo.M2;
        readYml["D2"] >> myInfo.D2;
        readYml["M3"] >> myInfo.M3;
        readYml["D3"] >> myInfo.D3;
        readYml.open(extrinsicDir, cv::FileStorage::READ);
        readYml["R1"] >> myInfo.R1;
        readYml["P1"] >> myInfo.P1;
        readYml["Rwc"] >> myInfo.RW2C;
        readYml["Twc"] >> myInfo.TW2C;
        readYml["R2"] >> myInfo.R2;
        readYml["P2"] >> myInfo.P2;
        readYml["Q"] >> myInfo.Q;
        readYml["R"] >> myInfo.R;
        readYml["T"] >> myInfo.T;
        readYml["S"] >> myInfo.S;
        readYml.release();
    }

    MatrixsInfo::MatrixsInfo(std::string calibrationFileDir) {
        cv::FileStorage readYml(calibrationFileDir, cv::FileStorage::READ);
        readYml["M1"] >> myInfo.M1;
        readYml["D1"] >> myInfo.D1;
        readYml["M2"] >> myInfo.M3;
        readYml["D2"] >> myInfo.D3;
        readYml["K"] >> myInfo.K;
        readYml["R"] >> myInfo.R;
        readYml["T"] >> myInfo.T;
        readYml.release();
    }

    const Info &MatrixsInfo::getInfo() {
        return myInfo;
    }
}// namespace SL