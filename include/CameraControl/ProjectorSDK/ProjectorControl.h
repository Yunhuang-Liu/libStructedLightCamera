/**
 * @file ProjectorControl.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  投影仪控制类
 * @version 0.1
 * @date 2022-5-9
 *
 * @copyright Copyright (c) 2022
 *
 */


#ifndef ProjectorControl_H
#define ProjectorControl_H

#include "./dlpc_common.h"
#include "./dlpc34xx.h"
#include "./dlpc347x_internal_patterns.h"
#include "./cypress_i2c.h"
#include <string>

#define FLASH_WRITE_BLOCK_SIZE            1024
#define FLASH_READ_BLOCK_SIZE             256

#define MAX_WRITE_CMD_PAYLOAD             (FLASH_WRITE_BLOCK_SIZE + 8)
#define MAX_READ_CMD_PAYLOAD              (FLASH_READ_BLOCK_SIZE  + 8)

/** \write buffer **/
static uint8_t s_WriteBuffer[MAX_WRITE_CMD_PAYLOAD];
/** \write buffer **/
static uint8_t s_ReadBuffer[MAX_READ_CMD_PAYLOAD];
/** \file pointer **/
static FILE* s_FilePointer;

class ProjectorControl{
public:
    /**
     * @brief 构造函数
     * @param projectorType 输入，投影仪类别
     */
    ProjectorControl(const DLPC34XX_ControllerDeviceId_e projectorType);
    /**
     * @brief 投影一次
     */
    void projecteOnce();
    /**
     * @brief 加载固件
     * @param firmWareAdress 输入，固件文件地址
     */
    void LoadFirmware(const std::string firmWareAdress);
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
};
#endif
