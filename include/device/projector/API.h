/*
 * API.h
 *
 * This module provides C callable APIs for each of the command supported by LightCrafter4500 platform and detailed in the programmer's guide.
 *
 * Copyright (C) 2013 Texas Instruments Incorporated - http://www.ti.com/
 * ALL RIGHTS RESERVED
 *
*/

#ifndef API_H
#define API_H

/* Bit masks. */
#define BIT0        0x01
#define BIT1        0x02
#define BIT2        0x04
#define BIT3        0x08
#define BIT4        0x10
#define BIT5        0x20
#define BIT6        0x40
#define BIT7        0x80
#define BIT8      0x0100
#define BIT9      0x0200
#define BIT10     0x0400
#define BIT11     0x0800
#define BIT12     0x1000
#define BIT13     0x2000
#define BIT14     0x4000
#define BIT15     0x8000
#define BIT16 0x00010000
#define BIT17 0x00020000
#define BIT18 0x00040000
#define BIT19 0x00080000
#define BIT20 0x00100000
#define BIT21 0x00200000
#define BIT22 0x00400000
#define BIT23 0x00800000
#define BIT24 0x01000000
#define BIT25 0x02000000
#define BIT26 0x04000000
#define BIT27 0x08000000
#define BIT28 0x10000000
#define BIT29 0x20000000
#define BIT30 0x40000000
#define BIT31 0x80000000

#define STAT_BIT_FLASH_BUSY     BIT3
#define HID_MESSAGE_MAX_SIZE    512

typedef struct _hidmessageStruct
{
    struct _hidhead
    {
        struct _packetcontrolStruct
        {
            unsigned char dest		:3; /* 0 - ProjCtrl; 1 - RFC; 7 - Debugmsg */
            unsigned char reserved	:2;
            unsigned char nack		:1; /* Command Handler Error */
            unsigned char reply	:1; /* Host wants a reply from device */
            unsigned char rw		:1; /* Write = 0; Read = 1 */
        }flags;
        unsigned char seq;
        unsigned short length;
    }head;
    union
    {
        unsigned short cmd;
        unsigned char data[HID_MESSAGE_MAX_SIZE];
    }text;
}hidMessageStruct;

typedef struct _readCmdData
{
    unsigned char I2CCMD;
    unsigned char CMD2;
    unsigned char CMD3;
    bool batchUpdateEnable;
    unsigned short len;
    char *name;
}CmdFormat;

typedef struct _rectangle
{
    unsigned short firstPixel;
    unsigned short firstLine;
    unsigned short pixelsPerLine;
    unsigned short linesPerFrame;
}rectangle;

typedef enum
{   
    SOURCE_SEL,
    PIXEL_FORMAT,
    CLK_SEL,
    CHANNEL_SWAP,
    FPD_MODE,
    POWER_CONTROL,
    FLIP_LONG,
    FLIP_SHORT,
    TPG_SEL,
    PWM_INVERT,
    LED_ENABLE,
    GET_VERSION,
    SW_RESET,
    STATUS_HW,
    STATUS_SYS,
    STATUS_MAIN,
    PWM_ENABLE,
    PWM_SETUP,
    PWM_CAPTURE_CONFIG,
    GPIO_CONFIG,
    LED_CURRENT,
    DISP_CONFIG,
    DISP_MODE,
    TRIG_OUT1_CTL,
    TRIG_OUT2_CTL,
    RED_LED_ENABLE_DLY,
    GREEN_LED_ENABLE_DLY,
    BLUE_LED_ENABLE_DLY,
    PAT_START_STOP,
    TRIG_IN1_CTL,
    TRIG_IN2_CTL,
    INVERT_DATA,
    PAT_CONFIG,
    MBOX_ADDRESS,
    MBOX_CONTROL,
    MBOX_DATA,
    SPLASH_LOAD,
    GPCLK_CONFIG,
    TPG_COLOR,
    PWM_CAPTURE_READ,
    I2C_PASSTHRU,
    PATMEM_LOAD_INIT_MASTER,
    PATMEM_LOAD_DATA_MASTER,
    PATMEM_LOAD_INIT_SLAVE,
    PATMEM_LOAD_DATA_SLAVE,
    BATCHFILE_NAME,
    BATCHFILE_EXECUTE,
    DELAY,
    DEBUG,
    I2C_CONFIG,
    CURTAIN_COLOR,
    VIDEO_CONT_SEL,
    READ_ERROR_CODE,
    READ_ERROR_MSG,
    READ_FRMW_VERSION,
    DMD_BLOCKS,
    DMD_IDLE,
    BL_STATUS,
    BL_SPL_MODE,
    BL_GET_MANID,
    BL_GET_DEVID,
    BL_GET_CHKSUM,
    BL_SET_SECTADDR,
    BL_SECT_ERASE,
    BL_SET_DNLDSIZE,
    BL_DNLD_DATA,
    BL_FLASH_TYPE,
    BL_CALC_CHKSUM,
    BL_PROG_MODE,
    BL_MASTER_SLAVE,
}LCR_CMD;

int LCR_SetInputSource(unsigned int source, unsigned int portWidth);
int LCR_GetInputSource(unsigned int *pSource, unsigned int *portWidth);
int LCR_SetPixelFormat(unsigned int format);
int LCR_GetPixelFormat(unsigned int *pFormat);
int LCR_SetPortClock(unsigned int clock);
int LCR_GetPortClock(unsigned int *pClock);
int LCR_SetDataChannelSwap(unsigned int port, unsigned int swap);
int LCR_GetDataChannelSwap(unsigned int Port, unsigned int *pSwap);
int LCR_SetFPD_Mode_Field(unsigned int PixelMappingMode, bool SwapPolarity, unsigned int FieldSignalSelect);
int LCR_GetFPD_Mode_Field(unsigned int *pPixelMappingMode, bool *pSwapPolarity, unsigned int *pFieldSignalSelect);
int LCR_SetPowerMode(unsigned char);
int LCR_GetPowerMode(bool *Standby);
int LCR_SetLongAxisImageFlip(bool);
bool LCR_GetLongAxisImageFlip();
int LCR_SetShortAxisImageFlip(bool);
bool LCR_GetShortAxisImageFlip();
int LCR_SetTPGSelect(unsigned int pattern);
int LCR_GetTPGSelect(unsigned int *pPattern);
int LCR_SetLEDPWMInvert(bool invert);
int LCR_GetLEDPWMInvert(bool *inverted);
int LCR_SetLedEnables(bool SeqCtrl, bool Red, bool Green, bool Blue);
int LCR_GetLedEnables(bool *pSeqCtrl, bool *pRed, bool *pGreen, bool *pBlue);
int LCR_GetVersion(unsigned int *pApp_ver, unsigned int *pAPI_ver, unsigned int *pSWConfig_ver, unsigned int *pSeqConfig_ver);
int LCR_SoftwareReset(void);
int LCR_GetStatus(unsigned char *pHWStatus, unsigned char *pSysStatus, unsigned char *pMainStatus);
int LCR_SetPWMEnable(unsigned int channel, bool Enable);
int LCR_GetPWMEnable(unsigned int channel, bool *pEnable);
int LCR_SetPWMConfig(unsigned int channel, unsigned int pulsePeriod, unsigned int dutyCycle);
int LCR_GetPWMConfig(unsigned int channel, unsigned int *pPulsePeriod, unsigned int *pDutyCycle);
int LCR_SetPWMCaptureConfig(unsigned int channel, bool enable, unsigned int sampleRate);
int LCR_GetPWMCaptureConfig(unsigned int channel, bool *pEnabled, unsigned int *pSampleRate);
int LCR_SetGPIOConfig(unsigned int pinNum, bool dirOutput, bool outTypeOpenDrain, bool pinState);
int LCR_GetGPIOConfig(unsigned int pinNum, bool *pDirOutput, bool *pOutTypeOpenDrain, bool *pState);
int LCR_GetLedCurrents(unsigned char *pRed, unsigned char *pGreen, unsigned char *pBlue);
int LCR_SetLedCurrents(unsigned char RedCurrent, unsigned char GreenCurrent, unsigned char BlueCurrent);
int LCR_SetDisplay(rectangle croppedArea, rectangle displayArea);
int LCR_GetDisplay(rectangle *pCroppedArea, rectangle *pDisplayArea);
int LCR_MemRead(unsigned int addr, unsigned int *readWord);
int LCR_MemWrite(unsigned int addr, unsigned int data);
int LCR_ValidatePatLutData(unsigned int *pStatus);
int LCR_SetPatternDisplayMode(bool external);
int LCR_GetPatternDisplayMode(bool *external);
int LCR_SetTrigOutConfig(unsigned int trigOutNum, bool invert, short rising, short falling);
int LCR_GetTrigOutConfig(unsigned int trigOutNum, bool *pInvert,short *pRising, short *pFalling);
int LCR_SetRedLEDStrobeDelay(short rising, short falling);
int LCR_SetGreenLEDStrobeDelay(short rising, short falling);
int LCR_SetBlueLEDStrobeDelay(short rising, short falling);
int LCR_GetRedLEDStrobeDelay(short *, short *);
int LCR_GetGreenLEDStrobeDelay(short *, short *);
int LCR_GetBlueLEDStrobeDelay(short *, short *);
int LCR_EnterProgrammingMode(void);
int LCR_ExitProgrammingMode(void);
int LCR_GetProgrammingMode(bool *ProgMode);
int LCR_EnableMasterSlave(void);
int LCR_DisableMasterUpdate(void);
int LCR_DisableSlaveUpdate(void);
int LCR_GetFlashManID(unsigned short *manID);
int LCR_GetFlashDevID(unsigned long long *devID);
int LCR_GetBLStatus(unsigned char *BL_Status);
int LCR_SetFlashAddr(unsigned int Addr);
int LCR_FlashSectorErase(void);
int LCR_SetDownloadSize(unsigned int dataLen);
int LCR_DownloadData(unsigned char *pByteArray, unsigned int dataLen);
void LCR_WaitForFlashReady(void);
int LCR_SetFlashType(unsigned char Type);
int LCR_CalculateFlashChecksum(void);
int LCR_GetFlashChecksum(unsigned int*checksum);
int LCR_SetMode(int SLmode);
int LCR_GetMode(int *pMode);
int LCR_LoadSplash(unsigned int index);
int LCR_GetSplashIndex(unsigned int *pIndex);
int LCR_SetTPGColor(unsigned short redFG, unsigned short greenFG, unsigned short blueFG, unsigned short redBG, unsigned short greenBG, unsigned short blueBG);
int LCR_GetTPGColor(unsigned short *pRedFG, unsigned short *pGreenFG, unsigned short *pBlueFG, unsigned short *pRedBG, unsigned short *pGreenBG, unsigned short *pBlueBG);
int LCR_ClearPatLut(void);
int LCR_AddToPatLut(int patNum, int ExpUs, bool ClearPat, int BitDepth, int LEDSelect, bool WaitForTrigger, int DarkTime, bool TrigOut2, int SplashIndex, int BitIndex);
int LCR_SendPatLut(void);
int LCR_SendSplashLut(unsigned char *lutEntries, unsigned int numEntries);
int LCR_GetPatLut(int numEntries);
int LCR_GetSplashLut(unsigned char *pLut, int numEntries);
int LCR_SetPatternTriggerMode(bool);
int LCR_GetPatternTriggerMode(bool *);
int LCR_PatternDisplay(int Action);
int LCR_SetPatternConfig(unsigned int numLutEntries, unsigned int repeat);
int LCR_GetPatternConfig(unsigned int *pNumLutEntries, bool *pRepeat, unsigned int *pNumPatsForTrigOut2, unsigned int *pNumSplash);
int LCR_SetTrigIn1Config(bool invert, unsigned int trigDelay);
int LCR_GetTrigIn1Config(bool *pInvert, unsigned int *pTrigDelay);
int LCR_SetTrigIn1Delay(unsigned int Delay);
int LCR_GetTrigIn1Delay(unsigned int *pDelay);
int LCR_SetTrigIn2Config(bool invert);
int LCR_GetTrigIn2Config(bool *pInvert);
int LCR_SetInvertData(bool invert);
int LCR_GetInvertData(bool *pInvert);
int LCR_PWMCaptureRead(unsigned int channel, unsigned int *pLowPeriod, unsigned int *pHighPeriod);
int LCR_SetGeneralPurposeClockOutFreq(unsigned int clkId, bool enable, unsigned int clkDivider);
int LCR_GetGeneralPurposeClockOutFreq(unsigned int clkId, bool *pEnabled, unsigned int *pClkDivider);
int LCR_MeasureSplashLoadTiming(unsigned int startIndex, unsigned int numSplash);
int LCR_ReadSplashLoadTiming(unsigned int *pTimingData);
int LCR_SetI2CPassThrough(unsigned int port, unsigned int addm, unsigned int clk, unsigned int devadd, unsigned char* wdata, unsigned int nwbytes);
int LCR_GetI2CPassThrough(unsigned int port, unsigned int addm, unsigned int clk, unsigned int devadd, unsigned char* wdata, unsigned int nwbytes, unsigned int nrbytes, unsigned char* rdata);
int LCR_WriteI2CPassThrough(unsigned int port, unsigned int devadd, unsigned char* wdata, unsigned int nwbytes);
int LCR_ReadI2CPassThrough(unsigned int port, unsigned int devadd, unsigned char* wdata, unsigned int nwbytes, unsigned int nrbytes, unsigned char* rdata);
int LCR_I2CConfigure(unsigned int port, unsigned int addm, unsigned int clk);
int LCR_SetPixelMode(unsigned int index);
int LCR_GetPixelMode(unsigned int *index);
int LCR_pattenMemLoad(bool master, unsigned char *pByteArray, int size);
int LCR_InitPatternMemLoad(bool master, unsigned short imageNum, unsigned int size);
int LCR_getBatchFileName(unsigned char id, char *batchFileName);
int LCR_executeBatchFile(unsigned char id);
int LCR_enableDebug();
int LCR_GetPortConfig(unsigned int *pDataPort,unsigned int *pPixelClock,unsigned int *pDataEnable,unsigned int *pSyncSelect);
int LCR_SetPortConfig(unsigned int dataPort,unsigned int pixelClock,unsigned int dataEnable,unsigned int syncSelect);
int LCR_executeRawCommand(unsigned char *rawCommand, int count);
void API_registerMainWindowCallback(void (*callback)(char *));
int API_getI2CCommand(char *command, unsigned char *i2cCommand);
int API_getUSBCommand(char *command, unsigned char *usbCommand);
int API_getCommandLength(char *command, int *len);
int API_getCommandName(unsigned char i2cCommand, char **command);
int API_getBatchFilePatternDetails(unsigned char *batchBuffer, int size, unsigned short *patternImageList, int *patternImageCount);
int API_changeImgNoinBatchFile(unsigned char *buffer, int size, int curId, int changeId);
int LCR_SetCurtainColor(unsigned int red,unsigned int green, unsigned int blue);
int LCR_GetCurtainColor(unsigned int *pRed, unsigned int *pGreen, unsigned int *pBlue);
int LCR_SetIT6535PowerMode(unsigned int powerMode);
int LCR_GetIT6535PowerMode(unsigned int *pPowerMode);
int LCR_ReadErrorCode(unsigned int *pCode);
int LCR_ReadErrorString(char *errStr);
int LCR_GetFrmwVersion(unsigned int *pFrmwType, char *pFrmwTag);
int LCR_SetDMDBlocks(int startBlock, int numBlocks);
int LCR_GetDMDBlocks(int *startBlock, int *numBlocks);
int LCR_SetDMDSaverMode(short mode);
int LCR_GetDMDSaverMode();

#endif // API_H
