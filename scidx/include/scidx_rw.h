/**
 *  @file scidx_rw.h
 *  @author Sheng Di
 *  @date Jan, 2023
 *  @brief Header file for the scidx_rw.cc
 *  (C) 2023 by 
 */

#ifndef _SCIDX_RW_H
#define _SCIDX_RW_H

#include <stdio.h>
#include <stdint.h>
#include <scidx_defines.h>

#ifdef __cplusplus
extern "C" {
#endif

int scidx_checkFileExistance(char* filePath);

size_t scidx_checkFileSize(char *srcFilePath, int *status);

unsigned char *scidx_readByteData(char *srcFilePath, size_t *byteLength, int *status);
double *scidx_readDoubleData(char *srcFilePath, size_t *nbEle, int *status);
float *scidx_readFloatData(char *srcFilePath, size_t *nbEle, int *status);

double *scidx_readDoubleData_systemEndian(char *srcFilePath, size_t *nbEle, int *status);
float *scidx_readFloatData_systemEndian(char *srcFilePath, size_t *nbEle, int *status);

void scidx_writeByteData(unsigned char *bytes, size_t byteLength, char *tgtFilePath, int *status);
void scidx_writeDoubleData(double *data, size_t nbEle, char *tgtFilePath, int *status);
void scidx_writeFloatData(float *data, size_t nbEle, char *tgtFilePath, int *status);
void scidx_writeIntData(int *data, size_t nbEle, char *tgtFilePath, int *status);
void scidx_writeData(void *data, int dataType, size_t nbEle, char *tgtFilePath, int *status);
void scidx_writeFloatData_inBytes(float *data, size_t nbEle, char* tgtFilePath, int *status);
void scidx_writeDoubleData_inBytes(double *data, size_t nbEle, char* tgtFilePath, int *status);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SCIDX_RW_H  ----- */
