/*
 * usb.cpp
 *
 * This module has the wrapper functions to access USB driver functions.
 *
 * Copyright (C) 2013 Texas Instruments Incorporated - http://www.ti.com/
 * ALL RIGHTS RESERVED
 *
*/


//#include "mainwindow.h"
//#include "ui_mainwindow.h"
//#ifdef Q_OS_WIN32
//#include <setupapi.h>
//#endif

#include <device/projector/usb.h>
#include <device/projector/hidapi.h>
#include <device/projector/API.h>

#include <stdio.h>
#include <stdlib.h>

/***************************************************
*                  GLOBAL VARIABLES
****************************************************/
static hid_device *DeviceHandle;	//Handle to write
//In/Out buffers equal to HID endpoint size + 1
//First byte is for Windows internal use and it is always 0
unsigned char OutputBuffer[USB_MAX_PACKET_SIZE+1];
unsigned char InputBuffer[USB_MAX_PACKET_SIZE+1];

static bool FakeConnection = false;
static bool USBConnected = false;      //Boolean true when device is connected

void USB_SetFakeConnection(bool enable)
{
    FakeConnection = enable;
}

bool USB_IsConnected()
{
    return USBConnected;
}

int USB_Init(void)
{
    return hid_init();
}

int USB_Exit(void)
{
    return hid_exit();
}

int USB_Open()
{
    if(FakeConnection == false)
    {
        // Open the device using the VID, PID,
        // and optionally the Serial number.
        DeviceHandle = hid_open(MY_VID, MY_PID, NULL);

        if(DeviceHandle == NULL)
        {
            USBConnected = false;
            return -1;
        }
    }
    USBConnected = true;
    return 0;
}

static hidMessageStruct dummyMsg;
static unsigned char powermode;
static unsigned char dispmode;

int USB_Write()
{
    if(FakeConnection == true)
    {
        memcpy(&dummyMsg, OutputBuffer + 1, 16);

        switch(dummyMsg.text.cmd)
        {
        case  0x200: // power mode
            if(dummyMsg.head.flags.rw == 1) // read
            {
                dummyMsg.text.data[0] = powermode;
                memcpy(InputBuffer, &dummyMsg, 16);
            }
            else
            {
                powermode = dummyMsg.text.data[2];
            }
            break;

        case 0x1A1B:
            if(dummyMsg.head.flags.rw == 1) // read
            {
                dummyMsg.text.data[0] = dispmode;
                memcpy(InputBuffer, &dummyMsg, 16);
            }
            else
            {
                dispmode = dummyMsg.text.data[2];
            }
            break;

        default:
            memset(InputBuffer, 0, 16);
        }
        return 1;
    }

    if(DeviceHandle == NULL)
        return -1;

    /*    for (int i = 0; i < USB_MIN_PACKET_SIZE; i++)
        printf("0x%x ", OutputBuffer[i]);
    printf("\n\n");*/
    return hid_write(DeviceHandle, OutputBuffer, USB_MIN_PACKET_SIZE+1);
}

int USB_Read()
{
    if(FakeConnection == true)
        return 1;

    if(DeviceHandle == NULL)
        return -1;

    return hid_read_timeout(DeviceHandle, InputBuffer, USB_MIN_PACKET_SIZE+1, 10000);
}

int USB_Close()
{
    if(FakeConnection == false)
    {
        hid_close(DeviceHandle);
        USBConnected = false;
    }
    return 0;
}
