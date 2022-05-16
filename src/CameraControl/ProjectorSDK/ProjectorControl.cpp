#include <CameraControl/ProjectorSDK/ProjectorControl.h>

/**
 * @brief I2C Write
 * @param WriteDataLength in        dalength
 * @param WriteData       in        data
 * @param ProtocolData    in        proto
 * @return
 */
uint32_t WriteI2C(uint16_t             WriteDataLength,
    uint8_t* WriteData,
    DLPC_COMMON_CommandProtocolData_s* ProtocolData){
    bool Status = true;
    //printf("Write I2C Starts, length %d!!! \n", WriteDataLength);
    Status = CYPRESS_I2C_WriteI2C(WriteDataLength, WriteData);
    if (Status != true)
    {
        //printf("Write I2C Error!!! \n");
        return FAIL;
    }

    return SUCCESS;
}
/**
 * @brief I2C Read
 * @param WriteDataLength in        datalength
 * @param WriteData       in/out    data
 * @param ReadDataLength  in        dataLength
 * @param ReadData        in/out    data
 * @param ProtocolData    in        proto
 * @return
 */
uint32_t ReadI2C(uint16_t              WriteDataLength,
    uint8_t* WriteData,
    uint16_t                           ReadDataLength,
    uint8_t* ReadData,
    DLPC_COMMON_CommandProtocolData_s* ProtocolData){
    bool Status = 0;
    //printf("Write/Read I2C Starts, length %d!!! \n", WriteDataLength);
    Status = CYPRESS_I2C_WriteI2C(WriteDataLength, WriteData);
    if (Status != true)
    {
        //printf("Write I2C Error!!! \n");
        return FAIL;
    }

    Status = CYPRESS_I2C_ReadI2C(ReadDataLength, ReadData);
    if (Status != true)
    {
        //printf("Read I2C Error!!! \n");
        return FAIL;
    }

    return SUCCESS;
}

ProjectorControl::ProjectorControl(const DLPC34XX_ControllerDeviceId_e projectorType){
    InitConnectionAndCommandLayer();
    bool Status = CYPRESS_I2C_RequestI2CBusAccess();
    DLPC34XX_ControllerDeviceId_e DeviceId = projectorType;
    DLPC34XX_ReadControllerDeviceId(&DeviceId);
    //load patter from flash
    DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_STOP, 0);
    LoadPatternOrderTableEntryfromFlash();
    //trigger once test
    DLPC34XX_WriteTriggerOutConfiguration(DLPC34XX_TT_TRIGGER1, DLPC34XX_TE_ENABLE, DLPC34XX_TI_NOT_INVERTED, 0);
    DLPC34XX_WriteTriggerOutConfiguration(DLPC34XX_TT_TRIGGER2, DLPC34XX_TE_ENABLE, DLPC34XX_TI_NOT_INVERTED, 0);
    DLPC34XX_WriteTriggerInConfiguration(DLPC34XX_TE_DISABLE, DLPC34XX_TP_ACTIVE_HI);
    DLPC34XX_WritePatternReadyConfiguration(DLPC34XX_TE_DISABLE, DLPC34XX_TP_ACTIVE_HI);
    DLPC34XX_WriteOperatingModeSelect(DLPC34XX_OM_SENS_INTERNAL_PATTERN);
    DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_START, 0x00);
}

void ProjectorControl::projecteOnce(){
    DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_START, 0x00);
}

/**
 * Initialize the command layer by setting up the read/write buffers and
 * callbacks.
 */
void ProjectorControl::InitConnectionAndCommandLayer()
{
    DLPC_COMMON_InitCommandLibrary(s_WriteBuffer,
        sizeof(s_WriteBuffer),
        s_ReadBuffer,
        sizeof(s_ReadBuffer),
        WriteI2C,
        ReadI2C);

    CYPRESS_I2C_ConnectToCyI2C();
}

void ProjectorControl::LoadPatternOrderTableEntryfromFlash()
{
    DLPC34XX_PatternOrderTableEntry_s PatternOrderTableEntry;

    /* Reload from Flash */
    DLPC34XX_WritePatternOrderTableEntry(DLPC34XX_WC_RELOAD_FROM_FLASH, &PatternOrderTableEntry);
}

void ProjectorControl::LoadFirmware(const std::string firmWareAdress)
{
    /* write up to 1024 bytes of data */
    uint8_t FlashDataArray[1024];

    /* Pattern File assumes to be in the \build\vs2017\dlpc343x folder */
    s_FilePointer = fopen(firmWareAdress.data(), "rb");
    if (!s_FilePointer)
    {
        printf("Error opening the flash image file!");
        return;
    }
    fseek(s_FilePointer, 0, SEEK_END);
    uint32_t FlashDataSize = ftell(s_FilePointer);
    fseek(s_FilePointer, 0, SEEK_SET);

    /* Select Flash Data Block and Erase the Block */
    DLPC34XX_WriteFlashDataTypeSelect(DLPC34XX_FDTS_ENTIRE_FLASH);
    DLPC34XX_WriteFlashErase();

    /* Read Short Status to make sure Erase is completed */
    DLPC34XX_ShortStatus_s ShortStatus;
    do
    {
        DLPC34XX_ReadShortStatus(&ShortStatus);
    } while (ShortStatus.FlashEraseComplete == DLPC34XX_FE_NOT_COMPLETE);

    DLPC34XX_WriteFlashDataLength(1024);
    fread(FlashDataArray, sizeof(FlashDataArray), 1, s_FilePointer);
    DLPC34XX_WriteFlashStart(1024, FlashDataArray);

    int32_t BytesLeft = FlashDataSize - 1024;
    do
    {
        fread(FlashDataArray, sizeof(FlashDataArray), 1, s_FilePointer);
        DLPC34XX_WriteFlashContinue(1024, FlashDataArray);

        BytesLeft = BytesLeft - 1024;
    } while (BytesLeft > 0);

    fclose(s_FilePointer);
}
