/**
 * @file MatrixsInfo.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2021-12-10
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef RESTRUCTOR_MATRIXINFO_H
#define RESTRUCTOR_MATRIXINFO_H

#include<string>

#include<opencv2/opencv.hpp>


/** @brief 信息结构体    */
struct Info{
    /** \左相机内参矩阵 **/
    cv::Mat M1;  
    /** \右相机内参矩阵 **/
    cv::Mat M2;  
    /** \彩色相机内参矩阵 **/
    cv::Mat M3;
    /** \左相机相机坐标系到右相机坐标系的旋转矩阵 **/
    cv::Mat R1;  
    /** \右相机相机坐标系到左相机坐标系的旋转矩阵 **/
    cv::Mat R2;  
    /** \左相机相机坐标系到右相机的投影矩阵 **/
    cv::Mat P1;  
    /** \右相机相机坐标系到世界坐标系的投影矩阵 **/
    cv::Mat P2;  
    /** \左相机的畸变矩阵 **/
    cv::Mat D1;  
    /** \右相机的畸变矩阵 **/
    cv::Mat D2;  
    /** \彩色相机的畸变矩阵 **/
    cv::Mat D3;
    /** \深度映射矩阵 **/
    cv::Mat Q;  
    /** \相位-高度映射矩阵 **/
    cv::Mat K;
    /** \左相机至彩色相机旋转矩阵 **/
    cv::Mat R;
    /** \左相机至彩色相机平移矩阵 **/
    cv::Mat T;
    /** \Base 2 Camera R **/
    cv::Mat RW2C;
    /** \Base 2 Camera T **/
    cv::Mat TW2C;
    /** \相机幅面 **/
    cv::Mat S;
};

/** @brief 相机标定信息类    */
class MatrixsInfo{
public: 
    /**
     * @brief 类的初始化，完成内外参数信息的导入
     * 
     * @param intrinsicsPath 输入，内参文件路径
     * @param extrinsicsPath 输入，外参文件路径
     */
    MatrixsInfo(std::string intrinsicsPath,std::string extrinsicsPath);
    /**
     * @brief 类的初始化，完成内外参数信息的导入
     *
     * @param intrinsicsPath 输入，系统参数文件路径
     */
    MatrixsInfo(std::string calibrationFileDir);
    /**
     * @brief 获取信息结构体
     * 
     * @return 返回信息结构体
    */
    const Info& getInfo();
private:
    /** \读取到的校正信息 **/
    Info myInfo;
};

#endif // RESTRUCTOR_MATRIXINFO_H