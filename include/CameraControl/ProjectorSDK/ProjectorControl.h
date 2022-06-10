/**
 * @file ProjectorControl.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  ͶӰ�ǿ�����
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
#include "./API.h"
#include "./usb.h"
#include <string>
#include <iostream>

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

/** \ͶӰ�ǿ����࣬Ӧ��ע����ǣ�DLP3010���õ�ȫ���֧�֣�����GUI���и���������DLP6500��������ʹ�㲻�ò�ʹ��GUI���Ƚ���ͼƬ������ **/
class ProjectorControl{
public:
    /**
     * @brief ���캯��
     * @param projectorType ���룬ͶӰ�����
     */
    ProjectorControl(const DLPC34XX_ControllerDeviceId_e projectorType);
    /** 
     * @brief ���캯��
     * @param projectorType ���룬ͶӰ�����
     */
    ProjectorControl(const int numLutEntries);
    /**
     * @brief ͶӰһ�� 
     * @param numLutEntries ���룬ͼƬ��Ŀ
     */
    void projecteOnce();
    /**
     * @brief ���ع̼�
     * @param firmWareAdress ���룬�̼��ļ���ַ
     */
    void LoadFirmware(const std::string firmWareAdress);
private:
    /**
     * @brief ��ʼ��I2C
     */
    void InitConnectionAndCommandLayer();
    /**
     * @brief �ȴ�
     * @param Seconds ���룬�ȴ�ʱ��
     */
    void WaitForSeconds(uint32_t Seconds);
    /**
     * @brief ���������ͼƬ
     */
    void LoadPatternOrderTableEntryfromFlash();
    /** \�Ƿ�ΪDLPC900����оƬ **/
    const bool isDLPC900;
};
#endif
