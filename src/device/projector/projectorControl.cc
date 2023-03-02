#include "../../../include/device/projector/projectorControl.h"

namespace sl {
    namespace device {
        /**
	 * Implement the I2C write transaction here. The sample code here sends
	 * data to the controller via the Cypress USB-Serial adapter.
	 */
	uint32_t WriteI2C(uint16_t                           WriteDataLength,
		          uint8_t*                           WriteData,
		          DLPC_COMMON_CommandProtocolData_s* ProtocolData)
	{
	    bool Status = true;
	    Status = CYPRESS_I2C_WriteI2C(WriteDataLength, WriteData);
	    if (Status != true)
	    {
		//printf("Write I2C Error!!! \n");
		return FAIL;
	    }

	    return SUCCESS;
	}

	/**
	 * Implement the I2C write/read transaction here. The sample code here
	 * receives data from the controller via the Cypress USB-Serial adapter.
	 */
	uint32_t ReadI2C(uint16_t                           WriteDataLength,
		         uint8_t*                           WriteData,
		         uint16_t                           ReadDataLength,
		         uint8_t*                           ReadData,
		         DLPC_COMMON_CommandProtocolData_s* ProtocolData)
	{
	    bool Status = 0;
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

	void WaitForSeconds(uint32_t Seconds)
	{
	    uint32_t retTime = (uint32_t)(time(0)) + Seconds;	// Get finishing time.
    	    while (time(0) < retTime);	
	}
	/**
	 * A sample function that generates a 1-bit (binary) 1-D pattern
	 * The function fills the byte array Data. Each byte in the in array corresponds
	 * to a pixel. For a 1-bit pattern the value of each byte should be 1 or 0.
	 */
	void PopulateOneBitPatternData(uint16_t Length, uint8_t* Data, uint16_t NumBars)
	{
	    uint16_t PixelPos  = 0;
	    uint16_t BarPos    = 0;
	    uint16_t BarWidth  = Length / NumBars;
	    uint8_t  PixelData = 0;

	    for (; PixelPos < Length; PixelPos++)
	    {
		Data[PixelPos] = PixelData;

		BarPos++;
		if (BarPos >= BarWidth)
		{
		    BarPos = 0;
		    PixelData = (PixelData == 0 ? 1 : 0);
		}
	    }
	}

	/**
	 * A sample function that generates an 8-bit (gray scale) 1-D pattern
	 * The function fills the byte array Data. Each byte in the in array corresponds
	 * to a pixel. For an 8-bit pattern the value of each byte can be 0 - 255.
	 */
	void PopulateEightBitPatternData(uint16_t Length, uint8_t* Data, uint16_t NumBars)
	{
	    uint16_t PixelPos     = 0;
	    uint16_t BarPos       = 0;
	    uint16_t BarWidth     = Length / (2 * NumBars);
	    uint8_t  PixelData    = 0;
	    uint16_t  PixelDataInc = (uint16_t)std::ceil(255.0 / BarWidth);

	    for (; PixelPos < Length; PixelPos++)
	    {
		Data[PixelPos] = PixelData;

		BarPos++;
		if (BarPos >= BarWidth)
		{
		    BarPos    = 0;
		    PixelDataInc = -PixelDataInc;
		}

		PixelData = (uint8_t)(PixelData + PixelDataInc);
	    }
	}

	/**
	 * Populates an array of DLPC34XX_INT_PAT_PatternSet_s
	 */
	void PopulatePatternSetData(uint16_t DMDWidth, uint16_t DMDHeight)
	{
	    uint8_t                        HorzPatternIdx = 0;
	    uint8_t                        VertPatternIdx = 0;
	    uint8_t                        PatternIdx     = 0;
	    uint8_t                        PatternSetIdx  = 0;
	    uint8_t                        Index;
	    uint16_t                       NumBars;
	    DLPC34XX_INT_PAT_PatternSet_s* PatternSet;

	    /* Create a 1-bit (binary) Horizontal Pattern Set */
	    PatternSet = &s_PatternSets[PatternSetIdx++];
	    PatternSet->BitDepth = DLPC34XX_INT_PAT_BITDEPTH_ONE;
	    PatternSet->Direction = DLPC34XX_INT_PAT_DIRECTION_HORIZONTAL;
	    PatternSet->PatternCount = NUM_ONE_BIT_HORIZONTAL_PATTERNS;
	    PatternSet->PatternArray = &s_Patterns[PatternIdx];
	    for (Index = 0; Index < NUM_ONE_BIT_HORIZONTAL_PATTERNS; Index++)
	    {
		NumBars = 2 * (Index + 1);
		PopulateOneBitPatternData(DMDHeight, s_HorizontalPatternData[HorzPatternIdx], NumBars);
		s_Patterns[PatternIdx].PixelArray = s_HorizontalPatternData[HorzPatternIdx];
		s_Patterns[PatternIdx].PixelArrayCount = DMDHeight;
		PatternIdx++;
		HorzPatternIdx++;
	    }

	    /* Create a 1-bit (binary) Vertical Pattern Set */
	    PatternSet = &s_PatternSets[PatternSetIdx++];
	    PatternSet->BitDepth = DLPC34XX_INT_PAT_BITDEPTH_ONE;
	    PatternSet->Direction = DLPC34XX_INT_PAT_DIRECTION_VERTICAL;
	    PatternSet->PatternCount = NUM_ONE_BIT_VERTICAL_PATTERNS;
	    PatternSet->PatternArray = &s_Patterns[PatternIdx];
	    for (Index = 0; Index < NUM_ONE_BIT_VERTICAL_PATTERNS; Index++)
	    {
		NumBars = 2 * (Index + 1);
		PopulateOneBitPatternData(DMDWidth, s_VerticalPatternData[VertPatternIdx], NumBars);
		s_Patterns[PatternIdx].PixelArray = s_VerticalPatternData[VertPatternIdx];
		s_Patterns[PatternIdx].PixelArrayCount = DMDWidth;
		PatternIdx++;
		VertPatternIdx++;
	    }

	    /* Create an 8-bit (grayscale) Horizontal Pattern Set */
	    PatternSet = &s_PatternSets[PatternSetIdx++];
	    PatternSet->BitDepth     = DLPC34XX_INT_PAT_BITDEPTH_EIGHT;
	    PatternSet->Direction    = DLPC34XX_INT_PAT_DIRECTION_HORIZONTAL;
	    PatternSet->PatternCount = NUM_EIGHT_BIT_HORIZONTAL_PATTERNS;
	    PatternSet->PatternArray = &s_Patterns[PatternIdx];
	    for (Index = 0; Index < NUM_EIGHT_BIT_HORIZONTAL_PATTERNS; Index++)
	    {
		NumBars = 2 * (Index + 1);
		PopulateEightBitPatternData(DMDHeight, s_HorizontalPatternData[HorzPatternIdx], NumBars);
		s_Patterns[PatternIdx].PixelArray      = s_HorizontalPatternData[HorzPatternIdx];
		s_Patterns[PatternIdx].PixelArrayCount = DMDHeight;
		PatternIdx++;
		HorzPatternIdx++;
	    }

	    /* Create an 8-bit (grayscale) Vertical Pattern Set */
	    PatternSet = &s_PatternSets[PatternSetIdx++];
	    PatternSet->BitDepth     = DLPC34XX_INT_PAT_BITDEPTH_EIGHT;
	    PatternSet->Direction    = DLPC34XX_INT_PAT_DIRECTION_VERTICAL;
	    PatternSet->PatternCount = NUM_EIGHT_BIT_VERTICAL_PATTERNS;
	    PatternSet->PatternArray = &s_Patterns[PatternIdx];
	    for (Index = 0; Index < NUM_EIGHT_BIT_VERTICAL_PATTERNS; Index++)
	    {
		NumBars = 2 * (Index + 1);
		PopulateEightBitPatternData(DMDWidth, s_VerticalPatternData[VertPatternIdx], NumBars);
		s_Patterns[PatternIdx].PixelArray      = s_VerticalPatternData[VertPatternIdx];
		s_Patterns[PatternIdx].PixelArrayCount = DMDWidth;
		PatternIdx++;
		VertPatternIdx++;
	    }
	}

	/**
	 * Populates an array of DLPC34XX_INT_PAT_PatternOrderTableEntry_s
	 */
	void PopulatePatternTableData()
	{
	    DLPC34XX_INT_PAT_PatternOrderTableEntry_s* PatternOrderTableEntry;
	    uint32_t                                   PatternOrderTableIdx = 0;
	    uint32_t                                   PatternSetIdx        = 0;

	    /* Pattern Table Entry 0 - uses Pattern Set 0 */
	    PatternOrderTableEntry = &s_PatternOrderTable[PatternOrderTableIdx++];
	    PatternOrderTableEntry->PatternSetIndex                        = PatternSetIdx;
	    PatternOrderTableEntry->NumDisplayPatterns                     = s_PatternSets[PatternSetIdx++].PatternCount;
	    PatternOrderTableEntry->IlluminationSelect                     = DLPC34XX_INT_PAT_ILLUMINATION_RED;
	    PatternOrderTableEntry->InvertPatterns                         = false;
	    PatternOrderTableEntry->IlluminationTimeInMicroseconds         = 5000;
	    PatternOrderTableEntry->PreIlluminationDarkTimeInMicroseconds  = 250;
	    PatternOrderTableEntry->PostIlluminationDarkTimeInMicroseconds = 1000;

	    /* Pattern Table Entry 1 - uses Pattern Set 1 */
	    PatternOrderTableEntry = &s_PatternOrderTable[PatternOrderTableIdx++];
	    PatternOrderTableEntry->PatternSetIndex                        = PatternSetIdx;
	    PatternOrderTableEntry->NumDisplayPatterns                     = s_PatternSets[PatternSetIdx++].PatternCount;
	    PatternOrderTableEntry->IlluminationSelect                     = DLPC34XX_INT_PAT_ILLUMINATION_GREEN;
	    PatternOrderTableEntry->InvertPatterns                         = false;
	    PatternOrderTableEntry->IlluminationTimeInMicroseconds         = 5000;
	    PatternOrderTableEntry->PreIlluminationDarkTimeInMicroseconds  = 250;
	    PatternOrderTableEntry->PostIlluminationDarkTimeInMicroseconds = 1000;

	    /* Pattern Table Entry 2 - uses Pattern Set 2 */
	    PatternOrderTableEntry = &s_PatternOrderTable[PatternOrderTableIdx++];
	    PatternOrderTableEntry->PatternSetIndex                        = PatternSetIdx;
	    PatternOrderTableEntry->NumDisplayPatterns                     = s_PatternSets[PatternSetIdx++].PatternCount;
	    PatternOrderTableEntry->IlluminationSelect                     = DLPC34XX_INT_PAT_ILLUMINATION_BLUE;
	    PatternOrderTableEntry->InvertPatterns                         = false;
	    PatternOrderTableEntry->IlluminationTimeInMicroseconds         = 5000;
	    PatternOrderTableEntry->PreIlluminationDarkTimeInMicroseconds  = 250;
	    PatternOrderTableEntry->PostIlluminationDarkTimeInMicroseconds = 1000;

	    /* Pattern Table Entry 3 - uses Pattern Set 3 */
	    PatternOrderTableEntry = &s_PatternOrderTable[PatternOrderTableIdx++];
	    PatternOrderTableEntry->PatternSetIndex                        = PatternSetIdx;
	    PatternOrderTableEntry->NumDisplayPatterns                     = s_PatternSets[PatternSetIdx++].PatternCount;
	    PatternOrderTableEntry->IlluminationSelect                     = DLPC34XX_INT_PAT_ILLUMINATION_RGB;
	    PatternOrderTableEntry->InvertPatterns                         = false;
	    PatternOrderTableEntry->IlluminationTimeInMicroseconds         = 11000;
	    PatternOrderTableEntry->PreIlluminationDarkTimeInMicroseconds  = 250;
	    PatternOrderTableEntry->PostIlluminationDarkTimeInMicroseconds = 1000;
	}

	void CopyDataToFlashProgramBuffer(uint8_t* Length, uint8_t** DataPtr)
	{
	    while ((*Length >= 1) && (s_FlashProgramBufferPtr < sizeof(s_FlashProgramBuffer)))
	    {
		s_FlashProgramBuffer[s_FlashProgramBufferPtr] = **DataPtr;
		s_FlashProgramBufferPtr++;
		(*DataPtr)++;
		(*Length)--;
	    }
	}

	void ProgramFlashWithDataInBuffer(uint16_t Length)
	{
	    s_FlashProgramBufferPtr = 0;

	    if (s_StartProgramming)
	    {
		s_StartProgramming = false;
		DLPC34XX_DUAL_WriteFlashStart(Length, s_FlashProgramBuffer);
	    }
	    else
	    {
		DLPC34XX_DUAL_WriteFlashContinue(Length, s_FlashProgramBuffer);
	    }
	}

	void WriteDataToFile(uint8_t Length, uint8_t* Data)
	{
	    fwrite(Data, 1, Length, s_FilePointer);
	}

	void GenerateAndWritePatternDataToFile(DLPC34XX_INT_PAT_DMD_e DMD, char* FilePath, bool EastWestFlip)
	{
	    s_FilePointer = fopen(FilePath, "wb");

	    /* Generate pattern data and write it to the flash.
	     * The DLPC34XX_INT_PAT_GeneratePatternDataBlock() function will call the
	     * WriteDataToFile() function several times while it packs sections of the
	     * pattern data.
	     */
	    DLPC34XX_INT_PAT_GeneratePatternDataBlock(DMD,
		                                      NUM_PATTERN_SETS,
		                                      s_PatternSets,
		                                      NUM_PATTERN_ORDER_TABLE_ENTRIES,
		                                      s_PatternOrderTable,
		                                      WriteDataToFile,
												  EastWestFlip);

	    fclose(s_FilePointer);
	}

	void BufferPatternDataAndProgramToFlash(uint8_t Length, uint8_t* Data)
	{
	    /* Copy data that can fit in the flash programming buffer */
	    CopyDataToFlashProgramBuffer(&Length, &Data);

	    /* Write data to flash if the buffer is full */
	    if (s_FlashProgramBufferPtr >= sizeof(s_FlashProgramBuffer))
	    {
		ProgramFlashWithDataInBuffer((uint16_t)sizeof(s_FlashProgramBuffer));
	    }

	    /* Copy remaining data (if any) to the flash programming buffer */
	    CopyDataToFlashProgramBuffer(&Length, &Data);
	}

	void GenerateAndProgramPatternData(DLPC34XX_INT_PAT_DMD_e DMD, bool EastWestFlip)
	{
	    s_StartProgramming = true;
	    s_FlashProgramBufferPtr = 0;

	    /* Let the controller know that we're going to program pattern data */
	    DLPC34XX_DUAL_WriteFlashDataTypeSelect(DLPC34XX_DUAL_FDTS_ENTIRE_SENS_PATTERN_DATA);

	    /* Erase the flash sectors that store pattern data */
	    DLPC34XX_DUAL_WriteFlashErase();

	    /* Read Short Status to make sure Erase is completed */
	    DLPC34XX_DUAL_ShortStatus_s ShortStatus;
	    do
	    {
		DLPC34XX_DUAL_ReadShortStatus(&ShortStatus);
	    } while (ShortStatus.FlashEraseComplete == DLPC34XX_DUAL_FE_NOT_COMPLETE);

	    /* To program the flash, send blocks of data of up to 1024 bytes
	     * to the controller at a time. Repeat the process until the entire
	     * data is programmed to the flash.
	     * Let the controller know the size of a data block that will be
	     * transferred at a time.
	     */
	    DLPC34XX_DUAL_WriteFlashDataLength(sizeof(s_FlashProgramBuffer));

	    /* Generate pattern data and program it to the flash.
	     *
	     * The DLPC34XX_INT_PAT_GeneratePatternDataBlock() function calls the
	     * BufferPatternDataAndProgramToFlash() function several times while it
	     * generates pattern data.
	     *
	     * The BufferPatternDataAndProgramToFlash() function buffers data received,
	     * programming the buffer content only when it is full. This is done in an
	     * effort to make flash writes more efficient, overall greatly reducing the
	     * time it takes to program the pattern data.
	     *
	     * After returning from the DLPC34XX_INT_PAT_GeneratePatternBlock() function,
	     * check if there is any data left in the buffer and program it. This needs
	     * to be done since the BufferPatternDataAndProgramToFlash() function only
	     * programs the buffer content if full.
	     */
	    DLPC34XX_INT_PAT_GeneratePatternDataBlock(DMD,
		                                      NUM_PATTERN_SETS,
		                                      s_PatternSets,
		                                      NUM_PATTERN_ORDER_TABLE_ENTRIES,
		                                      s_PatternOrderTable,
		                                      BufferPatternDataAndProgramToFlash,
		                                      EastWestFlip);
	    if (s_FlashProgramBufferPtr > 0)
	    {
		/* Resend the block size since it could be less than
		 * the previously specified size
		 */
		DLPC34XX_DUAL_WriteFlashDataLength(s_FlashProgramBufferPtr);

		ProgramFlashWithDataInBuffer(s_FlashProgramBufferPtr);
	    }
	}

	void LoadPatternOrderTableEntryfromFlash()
	{
	    DLPC34XX_DUAL_PatternOrderTableEntry_s PatternOrderTableEntry;

	    /* Reload from Flash */
	    DLPC34XX_DUAL_WritePatternOrderTableEntry(DLPC34XX_DUAL_WC_RELOAD_FROM_FLASH, &PatternOrderTableEntry);
	}


	void LoadPatternOrderTableEntry(uint8_t PatternSetIndex)
	{
	    DLPC34XX_DUAL_PatternOrderTableEntry_s PatternOrderTableEntry;

	    /* Set PatternOrderTableEntry to select specific Pattern Set and configure settings */
	    PatternOrderTableEntry.PatSetIndex = 1;
	    PatternOrderTableEntry.NumberOfPatternsToDisplay = PatternSetIndex;
	    PatternOrderTableEntry.RedIlluminator = DLPC34XX_DUAL_IE_DISABLE;
	    PatternOrderTableEntry.GreenIlluminator = DLPC34XX_DUAL_IE_DISABLE;
	    PatternOrderTableEntry.BlueIlluminator = DLPC34XX_DUAL_IE_ENABLE;
	    PatternOrderTableEntry.PatternInvertLsword = 0;
	    PatternOrderTableEntry.PatternInvertMsword = 0;
	    PatternOrderTableEntry.IlluminationTime = 2000;
	    PatternOrderTableEntry.PreIlluminationDarkTime = 250;
	    PatternOrderTableEntry.PostIlluminationDarkTime = 60;
	    DLPC34XX_DUAL_WritePatternOrderTableEntry(DLPC34XX_DUAL_WC_START, &PatternOrderTableEntry);
	}

	void WriteTestPatternGridLines()
	{
	    /* Write Input Image Size */
	    DLPC34XX_DUAL_WriteInputImageSize(DLP4710_WIDTH, DLP4710_HEIGHT);

	    /* Write Display Size */
	    DLPC34XX_DUAL_WriteDisplaySize(DLP4710_WIDTH, DLP4710_HEIGHT);

	    /* Write Grid Lines */
	    DLPC34XX_DUAL_GridLines_s GridLines;
	    GridLines.Border = DLPC34XX_DUAL_BE_ENABLE;
	    GridLines.BackgroundColor = DLPC34XX_DUAL_C_GREEN;
	    GridLines.ForegroundColor = DLPC34XX_DUAL_C_MAGENTA;
	    GridLines.HorizontalForegroundLineWidth = 0xF;
	    GridLines.HorizontalBackgroundLineWidth = 0xF;
	    GridLines.VerticalForegroundLineWidth = 0xF;
	    GridLines.VerticalBackgroundLineWidth = 0xF;
	    DLPC34XX_DUAL_WriteGridLines(&GridLines);
	    DLPC34XX_DUAL_WriteOperatingModeSelect(DLPC34XX_DUAL_OM_TEST_PATTERN_GENERATOR);
	    WaitForSeconds(5);
	}

	void WriteLookSelect(uint8_t LookNumber)
	{
	    /* Read Current Operating Mode Selected */
	    DLPC34XX_DUAL_OperatingMode_e OperatingMode;
	    DLPC34XX_DUAL_ReadOperatingModeSelect(&OperatingMode);

	    /* Write RGB LED Current (based on Flash data) */
	    DLPC34XX_DUAL_WriteRgbLedCurrent(0x03E8, 0x03E8, 0x03E8);

	    /* Select Look */
	    DLPC34XX_DUAL_WriteLookSelect(LookNumber);

	    /* Submit Write Splash Screen Execute if in Splash Mode */
	    if ((OperatingMode == DLPC34XX_DUAL_OM_SPLASH_SCREEN ) ||
		(OperatingMode == DLPC34XX_DUAL_OM_SENS_SPLASH_PATTERN))
	    {
		DLPC34XX_DUAL_WriteSplashScreenExecute();
		WaitForSeconds(5);
	    }
	    WaitForSeconds(5);
	}

	void LoadPreBuildPatternData()
	{
	    /* write up to 1024 bytes of data */
	    uint8_t PatternDataArray[1024];

	    /* Pattern File assumes to be in the \build\vs2017\dlpc347x folder */
	    s_FilePointer = fopen("pattern_data_dual_gui.bin", "rb");
	    if (!s_FilePointer)
	    {
		//printf("Error opening the binary file!");
		return;
	    }
	    fseek(s_FilePointer, 0, SEEK_END);
	    uint32_t PatternDataSize = ftell(s_FilePointer);
	    fseek(s_FilePointer, 0, SEEK_SET);

	    /* Select Flash Data Block and Erase the Block */
	    DLPC34XX_DUAL_WriteFlashDataTypeSelect(DLPC34XX_DUAL_FDTS_ENTIRE_SENS_PATTERN_DATA);
	    DLPC34XX_DUAL_WriteFlashErase();

	    /* Read Short Status to make sure Erase is completed */
	    DLPC34XX_DUAL_ShortStatus_s ShortStatus;
	    do
	    {
		DLPC34XX_DUAL_ReadShortStatus(&ShortStatus);
	    } while (ShortStatus.FlashEraseComplete == DLPC34XX_DUAL_FE_NOT_COMPLETE);

	    DLPC34XX_DUAL_WriteFlashDataLength(1024);
	    fread(PatternDataArray, sizeof(PatternDataArray), 1, s_FilePointer);
	    DLPC34XX_DUAL_WriteFlashStart(1024, PatternDataArray);

	    int32_t BytesLeft = PatternDataSize - 1024;
	    do
	    {
		fread(PatternDataArray, sizeof(PatternDataArray), 1, s_FilePointer);
		DLPC34XX_DUAL_WriteFlashContinue(1024, PatternDataArray);

		BytesLeft = BytesLeft - 1024;
	    } while (BytesLeft > 0);

	    fclose(s_FilePointer);
	}

        ProjectorControl::ProjectorControl(const DLPC34XX_ControllerDeviceId_e projectorType, const int numLutEntries) : isDLPC900(false), elementSize(numLutEntries){
            CyLibraryInit();
            InitConnectionAndCommandLayer();
            CYPRESS_I2C_RequestI2CBusAccess();
            /*
            uint16_t PixelsPerLine, LinesPerFrame;
            DLPC34XX_DUAL_ReadInputImageSize(&PixelsPerLine, &LinesPerFrame);
            WriteLookSelect(0);
            WriteTestPatternGridLines();
            */
            //LoadPreBuildPatternData();
            DLPC34XX_DUAL_WriteRgbLedEnable(true, true, true);
            DLPC34XX_DUAL_WriteRgbLedMaxCurrent(12000, 16000, 16000);
            DLPC34XX_DUAL_WriteRgbLedCurrent(12000, 16000, 16000);
            DLPC34XX_DUAL_WriteTriggerOutConfiguration(DLPC34XX_DUAL_TT_TRIGGER1, DLPC34XX_DUAL_TE_ENABLE, DLPC34XX_DUAL_TI_NOT_INVERTED, 0);
            DLPC34XX_DUAL_WriteTriggerOutConfiguration(DLPC34XX_DUAL_TT_TRIGGER2, DLPC34XX_DUAL_TE_ENABLE, DLPC34XX_DUAL_TI_NOT_INVERTED, 0);
            DLPC34XX_DUAL_WriteTriggerInConfiguration(DLPC34XX_DUAL_TE_DISABLE, DLPC34XX_DUAL_TP_ACTIVE_HI);
            DLPC34XX_DUAL_WritePatternReadyConfiguration(DLPC34XX_DUAL_TE_DISABLE, DLPC34XX_DUAL_TP_ACTIVE_HI);
            LoadPatternOrderTableEntryfromFlash();
            DLPC34XX_DUAL_WriteOperatingModeSelect(DLPC34XX_DUAL_OM_SENS_INTERNAL_PATTERN);
            //DLPC34XX_DUAL_WriteInternalPatternControl(DLPC34XX_DUAL_PC_START, 0x00);
            /*
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
            */
        }

        ProjectorControl::ProjectorControl(const int numLutEntries) : isDLPC900(true), elementSize(numLutEntries) {
            USB_Init();
            int SLmode = 0;
            bool standBy = 0;

            if (USB_IsConnected() == false) {
                USB_Open();
            }
            /*
            if (LCR_SetPowerMode(0) < 0)
                std::cout << "DLPC900 Error:Unable to power on the board" << std::endl;
            if (LCR_SetMode(0x1) < 0)
                std::cout << "DLPC900 Error:Unable to switch to pattern mode" << std::endl;
            */
            
            LCR_SetLedCurrents(255, 255, 255);
        }

        void ProjectorControl::projecte(const bool isContinues) {
            if (!isDLPC900) {
           	if (isContinues)
	            DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_START, 0xFF);
	        else
	            DLPC34XX_WriteInternalPatternControl(DLPC34XX_PC_START, 0x00);
	        
            }
            else {
                if (isContinues)
                    LCR_SetPatternConfig(elementSize, 0);
                else
                    LCR_SetPatternConfig(elementSize, elementSize);
                if (LCR_PatternDisplay(0x2) < 0)
                    printf("Unable to stat pattern display \n");
            }
        }

        void ProjectorControl::stopProject() {
            if (!isDLPC900) {
                DLPC34XX_DUAL_WriteInternalPatternControl(DLPC34XX_DUAL_PC_STOP, 0);
            }
            else {
            	if (LCR_PatternDisplay(0x0) < 0)
                    printf("Unable to stop pattern display \n");
            }
        }

        /**
         * Initialize the command layer by setting up the read/write buffers and
         * callbacks.
         */
        void ProjectorControl::InitConnectionAndCommandLayer() {
            DLPC_COMMON_InitCommandLibrary(s_WriteBuffer,
                                   sizeof(s_WriteBuffer),
                                   s_ReadBuffer,
                                   sizeof(s_ReadBuffer),
                                   WriteI2C,
                                   ReadI2C);

    	    CYPRESS_I2C_ConnectToCyI2C();
        }

        void ProjectorControl::LoadPatternOrderTableEntryfromFlash() {
            DLPC34XX_PatternOrderTableEntry_s PatternOrderTableEntry;

            /* Reload from Flash */
            DLPC34XX_WritePatternOrderTableEntry(DLPC34XX_WC_RELOAD_FROM_FLASH, &PatternOrderTableEntry);
        }

        void ProjectorControl::LoadFirmware(const std::string firmWareAdress) {
            /* write up to 1024 bytes of data */
            uint8_t FlashDataArray[1024];

            /* Pattern File assumes to be in the \build\vs2017\dlpc343x folder */
            s_FilePointer = fopen(firmWareAdress.data(), "rb");
            if (!s_FilePointer) {
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
            do {
                DLPC34XX_ReadShortStatus(&ShortStatus);
            } while (ShortStatus.FlashEraseComplete == DLPC34XX_FE_NOT_COMPLETE);

            DLPC34XX_WriteFlashDataLength(1024);
            fread(FlashDataArray, sizeof(FlashDataArray), 1, s_FilePointer);
            DLPC34XX_WriteFlashStart(1024, FlashDataArray);

            int32_t BytesLeft = FlashDataSize - 1024;
            do {
                fread(FlashDataArray, sizeof(FlashDataArray), 1, s_FilePointer);
                DLPC34XX_WriteFlashContinue(1024, FlashDataArray);

                BytesLeft = BytesLeft - 1024;
            } while (BytesLeft > 0);

            fclose(s_FilePointer);
        }

        void ProjectorControl::loadPatternData(const std::string fileName) {
            /* write up to 1024 bytes of data */
            uint8_t PatternDataArray[1024];

            /* Pattern File assumes to be in the \build\vs2017\dlpc347x folder */
            s_FilePointer = fopen(fileName.data(), "rb");
            if (!s_FilePointer)
            {
                //printf("Error opening the binary file!");
                return;
            }
            fseek(s_FilePointer, 0, SEEK_END);
            uint32_t PatternDataSize = ftell(s_FilePointer);
            fseek(s_FilePointer, 0, SEEK_SET);

            /* Select Flash Data Block and Erase the Block */
            DLPC34XX_DUAL_WriteFlashDataTypeSelect(DLPC34XX_DUAL_FDTS_ENTIRE_SENS_PATTERN_DATA);
            DLPC34XX_DUAL_WriteFlashErase();

            /* Read Short Status to make sure Erase is completed */
            DLPC34XX_DUAL_ShortStatus_s ShortStatus;
            do
            {
                DLPC34XX_DUAL_ReadShortStatus(&ShortStatus);
            } while (ShortStatus.FlashEraseComplete == DLPC34XX_DUAL_FE_NOT_COMPLETE);

            DLPC34XX_DUAL_WriteFlashDataLength(1024);
            fread(PatternDataArray, sizeof(PatternDataArray), 1, s_FilePointer);
            DLPC34XX_DUAL_WriteFlashStart(1024, PatternDataArray);

            int32_t BytesLeft = PatternDataSize - 1024;
            do
            {
                fread(PatternDataArray, sizeof(PatternDataArray), 1, s_FilePointer);
                DLPC34XX_DUAL_WriteFlashContinue(1024, PatternDataArray);

                BytesLeft = BytesLeft - 1024;
            } while (BytesLeft > 0);

            fclose(s_FilePointer);
        }
    }// namespace device
}// namespace sl
