/**
 * Macro definition
 * Authors: Ning Yang, Lipeng Wan, Sheng Di
 *
**/

#include <sys/time.h>      /* For gettimeofday(), in microseconds */
#include <time.h>          /* For time(), in seconds */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#ifndef _SCIDX_DEFINES_H
#define _SCIDX_DEFINES_H

#ifdef __cplusplus
extern "C" {
#endif

#define scidx_VERNUM 0x0200
#define scidx_VER_MAJOR 1
#define scidx_VER_MINOR 1
#define scidx_VER_BUILD 0
#define scidx_VER_REVISION 0

#define scidx_FLOAT 0
#define scidx_DOUBLE 1
#define scidx_UINT8 2
#define scidx_INT8 3
#define scidx_UINT16 4
#define scidx_INT16 5
#define scidx_UINT32 6
#define scidx_INT32 7
#define scidx_UINT64 8
#define scidx_INT64 9

#define LITTLE_ENDIAN_DATA 0 //refers to the endian type of the data read from the disk
#define BIG_ENDIAN_DATA 1 //big_endian (ppc, max, etc.) ; little_endian (x86, x64, etc.)

#define LITTLE_ENDIAN_SYSTEM 0 //refers to the endian type of the system
#define BIG_ENDIAN_SYSTEM 1

//SUCCESS returning status
#define scidx_SCES 0  //successful
#define scidx_NSCS -1 //Not successful
#define scidx_FERR -2 //Failed to open input file
#define scidx_TERR -3 //wrong data type (should be only float or double)
#define scidx_DERR -4 //dimension error

typedef union lint16
{
	unsigned short usvalue;
	short svalue;
	unsigned char byte[2];
} lint16;

typedef union lint32
{
	int ivalue;
	unsigned int uivalue;
	unsigned char byte[4];
} lint32;

typedef union lint64
{
	long lvalue;
	unsigned long ulvalue;
	unsigned char byte[8];
} lint64;

typedef union ldouble
{
    double value;
    unsigned long lvalue;
    unsigned char byte[8];
} ldouble;

typedef union lfloat
{
    float value;
    unsigned int ivalue;
    unsigned char byte[4];
} lfloat;

extern int scidx_versionNumber[4];
	
extern int scidx_dataEndianType; //*endian type of the data read from disk
extern int scidx_sysEndianType; //*sysEndianType is actually set automatically.

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SCIDX_DEFINES_H  ----- */
