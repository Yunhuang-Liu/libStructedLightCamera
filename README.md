# **LibStructedCamera**
***
StructedCamera SDK to easy use.Mutiply solve phase algorithm and accelerate method can be used,  
ofcause you can choose only cpu can be used if you don't have a NVIDIA GPU.  
# Related Works
***
Please cite this work if you make use of this system in any of your own endeavors:  
> **@File：libStructedCamera**  
> **@Authour: Liu Yunhuang**  
> **@Email: 1369215984@qq.com**  
> **@Data: 2022.5.16**  
# 1. What do I need to build it?  
## 1.1 Windows  
***
- CMake  
- AVX  
- Eigen  
- OpenCV  
- CUDA(optional)  
Firstly,please make sure your cpu is supported by AVX.  
Then install cmake,Eigen and OpenCV(built with Eigen、CUDA).  
If you want to a higher performance, make sure your computer have a NVIDIA GPU and install CUDA.  
Finally,cmake, built and install it.  
## 1.2 Ubuntu  
**as same as windows install step,but you shoud install usbseral SDK and huaray SDK which based on the Linux system.  
because of that our code is only provid Windows lib kit.**  
# 2. How do I use it?  
***
You can find a StructedlightCameraConfig.cmake file in your install path.  
So you only to add your install path to your system path.  
Then,your CMakeLists.txt can add these:  
> find_package(StrutedLightCamera REQUIRED)  
> include_directory(${StructedLightCamera_INCLUDE_DIRS})  
> target_link_libraries(${StructedLightCamera_Libs})  
**Please be aware of that you should add CUDA defination to use CUDA.Like this:**  
> find_package(CUDA REQUIRED)  
> if(CUDA_FOUND)  
> add_defination(-DCUDA)  
> endif()  
The system is contrusted like this:  
![StructedLightCamera System](./StructedLightCamera.png)  
**You can be aware of that our phasesolve and restructor is a Factory mode,so you can get what you want though Polymorphism.**  
***  
**A example is like this:**  
> //Load calibration file  
> MatrixsInfo* matrixInfo = new MatrixsInfo("../systemFile/calibrationFiles/intrinsic.yml","../systemFile/  calibrationFiles/extrinsic.yml");  
> const Info& calibrationInfo = matrixInfo->getInfo();  
> //Choose gpu  
> cv::cuda::setDevice(0);  
> //Load ref img if use devidedspace phasesolver  
> cv::Mat leftRef = cv::imread("../systemFile/refImg/left.tif", cv::IMREAD_UNCHANGED);  
> cv::Mat rightRef = cv::imread("../systemFile/refImg/right.tif", cv::IMREAD_UNCHANGED);  
> //set params optional  
> RestructorType::RestructParamater params(-500, 500, 150, 350, 16);  
> params.block = dim3(32, 16, 1);  
> //construct our StructedLightCamera  
> StructedLightCamera* camera = new StructedLightCamera(calibrationInfo, StructedLightCamera::DevidedSpaceTimeMulUsed, StructedLightCamera::GPU ,params, leftRef, rightRef);  
> //set camera exposure time optional  
> camera->setExposureTime(3000, 20400);  
> //get one frame  
> std::vector<cv::cuda::GpuMat> depthImgs;  
> std::vector<cv::cuda::GpuMat> colorImgs;  
> camera->getOneFrame(depthImgs, colorImgs);  


