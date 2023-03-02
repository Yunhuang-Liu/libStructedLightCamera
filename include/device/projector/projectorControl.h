/**
 * @file projectorControl.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  投影仪控制类
 * @version 0.1
 * @date 2022-5-9
 *
 * @copyright Copyright (c) 2022
 *
 */


#ifndef PROJECTOR_PROJECTORCONTROL_H_
#define PROJECTOR_PROJECTORCONTROL_H_

#include "dlpc_common.h"
#include "dlpc34xx.h"
#include "dlpc34xx_dual.h"
#include "dlpc347x_internal_patterns.h"
#include "cypress_i2c.h"
#include "API.h"
#include "usb.h"
#include "CyUSBSerial.h"
#include "math.h"
#include "functional"

#include <string>
#include <iostream>
#include <thread>

#define MAX_WIDTH                         DLP4710_WIDTH
#define MAX_HEIGHT                        DLP4710_HEIGHT

#define NUM_PATTERN_SETS                  4
#define NUM_PATTERN_ORDER_TABLE_ENTRIES   4
#define NUM_ONE_BIT_HORIZONTAL_PATTERNS   4
#define NUM_EIGHT_BIT_HORIZONTAL_PATTERNS 4
#define NUM_ONE_BIT_VERTICAL_PATTERNS     4
#define NUM_EIGHT_BIT_VERTICAL_PATTERNS   4
#define TOTAL_HORIZONTAL_PATTERNS         (NUM_ONE_BIT_HORIZONTAL_PATTERNS + NUM_EIGHT_BIT_HORIZONTAL_PATTERNS)
#define TOTAL_VERTICAL_PATTERNS           (NUM_ONE_BIT_VERTICAL_PATTERNS + NUM_EIGHT_BIT_VERTICAL_PATTERNS)

#define FLASH_WRITE_BLOCK_SIZE            1024
#define FLASH_READ_BLOCK_SIZE             256

#define MAX_WRITE_CMD_PAYLOAD             (FLASH_WRITE_BLOCK_SIZE + 8)
#define MAX_READ_CMD_PAYLOAD              (FLASH_READ_BLOCK_SIZE  + 8)

static uint8_t                                   s_HorizontalPatternData[TOTAL_HORIZONTAL_PATTERNS][MAX_HEIGHT];
static uint8_t                                   s_VerticalPatternData[TOTAL_VERTICAL_PATTERNS][MAX_WIDTH];
static DLPC34XX_INT_PAT_PatternData_s            s_Patterns[TOTAL_HORIZONTAL_PATTERNS + TOTAL_VERTICAL_PATTERNS];
static DLPC34XX_INT_PAT_PatternSet_s             s_PatternSets[NUM_PATTERN_SETS];
static DLPC34XX_INT_PAT_PatternOrderTableEntry_s s_PatternOrderTable[NUM_PATTERN_ORDER_TABLE_ENTRIES];

static uint8_t                                   s_WriteBuffer[MAX_WRITE_CMD_PAYLOAD];
static uint8_t                                   s_ReadBuffer[MAX_READ_CMD_PAYLOAD];

static bool                                      s_StartProgramming;
static uint8_t                                   s_FlashProgramBuffer[FLASH_WRITE_BLOCK_SIZE];
static uint16_t                                  s_FlashProgramBufferPtr;

static FILE*                                     s_FilePointer;

/** @brief 结构光库 */
namespace sl {
    /** @brief 设备控制库 */
    namespace device {
        /**
         * @brief 投影仪控制类
         * @note 应当注意的是，DLP3010将得到全面的支持，无需GUI进行辅助，DLP6500的需求将迫使你不得不使用GUI事先进行图片的烧入 
         */
        class ProjectorControl {
        public:
            /**
             * @brief 构造函数
             * @param projectorType 输入，投影仪类别
             */
            ProjectorControl(const DLPC34XX_ControllerDeviceId_e projectorType, const int numLutEntries);
            /** 
             * @brief 构造函数
             * @param projectorType 输入，投影仪类别
             */
            ProjectorControl(const int numLutEntries);
            /**
             * @brief 开始投影
             * @param isContinues 输入，是否连续投影
             */
            void projecte(const bool isContinues);
            /**
             * @brief 停止投影
             */
            void stopProject();
            /**
             * @brief 加载固件
             * @param firmWareAdress 输入，固件文件地址
             */
            void LoadFirmware(const std::string firmWareAdress);
            /**
             * @brief 加载预先存储的图片数据
             * @param fileName  输入，文件名(*.bin)
             */
            void loadPatternData(const std::string fileName);
            /** \投影图片张数 **/
            const int elementSize;
        private:
            /**
             * @brief 初始化I2C
             */
            void InitConnectionAndCommandLayer();
            /**
             * @brief 等待
             * @param Seconds 输入，等待时长
             */
            void WaitForSeconds(uint32_t Seconds);
            /**
             * @brief 从闪存加载图片
             */
            void LoadPatternOrderTableEntryfromFlash();
            /** \是否为DLPC900控制芯片 **/
            const bool isDLPC900;
        };
    }// namespace device
}// namespace sl
#endif //PROJECTOR_PROJECTORCONTROL_H_
