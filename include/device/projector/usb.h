/*
 * usb.h
 *
 * This module has the wrapper functions to access USB driver functions.
 *
 * Copyright (C) 2013 Texas Instruments Incorporated - http://www.ti.com/
 * ALL RIGHTS RESERVED
 *
*/

#ifndef USB_H
#define USB_H

#include <cstring>

#define USB_MIN_PACKET_SIZE 64
#define USB_MAX_PACKET_SIZE 64

#define MY_VID 0x0451
//#define MY_PID 0xC900
#define MY_PID 0x4421

void USB_SetFakeConnection(bool enable);
int USB_Open(void);
bool USB_IsConnected();
int USB_Write();
int USB_Read();
int USB_Close();
int USB_Init();
int USB_Exit();


#endif //USB_H
