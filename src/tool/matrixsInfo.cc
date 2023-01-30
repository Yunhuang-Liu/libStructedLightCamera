#include <tool/matrixsInfo.h>

namespace sl {
    namespace tool {
        MatrixsInfo::MatrixsInfo(std::string intrinsicDir, std::string extrinsicDir) {
            cv::FileStorage readYml(intrinsicDir, cv::FileStorage::READ);
            readYml["M1"] >> myInfo.M1;
            readYml["D1"] >> myInfo.D1;
            readYml["M2"] >> myInfo.M2;
            readYml["D2"] >> myInfo.D2;
            readYml["M3"] >> myInfo.M3;
            readYml["D3"] >> myInfo.D3;
            readYml["M4"] >> myInfo.M4;
            readYml["D4"] >> myInfo.D4;
            readYml["K1"] >> myInfo.K1;
            readYml["K2"] >> myInfo.K2;
            readYml.open(extrinsicDir, cv::FileStorage::READ);
            readYml["R1"] >> myInfo.R1;
            readYml["P1"] >> myInfo.P1;
            readYml["Rwc"] >> myInfo.RW2C;
            readYml["Twc"] >> myInfo.TW2C;
            readYml["R2"] >> myInfo.R2;
            readYml["P2"] >> myInfo.P2;
            readYml["Q"] >> myInfo.Q;
            readYml["Rlr"] >> myInfo.Rlr;
            readYml["Tlr"] >> myInfo.Tlr;
            readYml["Rlc"] >> myInfo.Rlc;
            readYml["Tlc"] >> myInfo.Tlc;
            readYml["Rlp"] >> myInfo.Rlp;
            readYml["Tlp"] >> myInfo.Tlp;
            readYml["Rrp"] >> myInfo.Rrp;
            readYml["Trp"] >> myInfo.Trp;
            readYml["S"] >> myInfo.S;
            readYml.release();
        }

        MatrixsInfo::MatrixsInfo(std::string calibrationFileDir) {
            cv::FileStorage readYml(calibrationFileDir, cv::FileStorage::READ);
            readYml["M1"] >> myInfo.M1;
            readYml["D1"] >> myInfo.D1;
            readYml["M2"] >> myInfo.M2;
            readYml["D2"] >> myInfo.D2;
            readYml["K1"] >> myInfo.K1;
            readYml["Rlr"] >> myInfo.Rlr;
            readYml["Tlr"] >> myInfo.Tlr;
            readYml.release();
        }

        const Info &MatrixsInfo::getInfo() {
            return myInfo;
        }
    }// namespace tool
}// namespace sl