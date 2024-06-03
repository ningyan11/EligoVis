/**
 *  @file scidx_rw.c
 *  @author Sheng Di
 *  @date Dec, 2023
 *  @brief io interface for fortrance
 *  (C) 2023 by 
 */


#include <scidx_avl_interval_tree.h>
#include <scidx_rb_interval_tree.h>
#include "scidx_rw.h"
#include <scidx_defines.h>
#include "scidx_BytesToolkit.h"

using namespace scidx;

int scidx_checkFileExistance(char* filePath)
{
	if( access( filePath, F_OK ) != -1 ) {
		// file exists
		return 1;
	} else {
		// file doesn't exist
		return 0;
	}	
}

size_t scidx_checkFileSize(char *srcFilePath, int *status)
{
	size_t filesize;
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
	{
		printf("Failed to open input file. 1\n");
		*status = scidx_FERR;
		return -1;
	}
	fseek(pFile, 0, SEEK_END);
    filesize = ftell(pFile);
    fclose(pFile);
    *status = scidx_SCES;
    return filesize;
}

unsigned char *scidx_readByteData(char *srcFilePath, size_t *byteLength, int *status)
{
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 1\n");
        *status = scidx_FERR;
        return 0;
    }
	fseek(pFile, 0, SEEK_END);
    *byteLength = ftell(pFile);
    fclose(pFile);
    
    unsigned char *byteBuf = ( unsigned char *)malloc((*byteLength)*sizeof(unsigned char)); //sizeof(char)==1
    
    pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 2\n");
        *status = scidx_FERR;
        return 0;
    }
    fread(byteBuf, 1, *byteLength, pFile);
    fclose(pFile);
    *status = scidx_SCES;
    return byteBuf;
}

double *scidx_readDoubleData(char *srcFilePath, size_t *nbEle, int *status)
{
	int state = scidx_SCES;
	if(scidx_dataEndianType==scidx_sysEndianType)
	{
		double *daBuf = scidx_readDoubleData_systemEndian(srcFilePath, nbEle,&state);
		*status = state;
		return daBuf;
	}
	else
	{
		size_t i,j;
		
		size_t byteLength;
		unsigned char* bytes = scidx_readByteData(srcFilePath, &byteLength, &state);
		if(state==scidx_FERR)
		{
			*status = scidx_FERR;
			return NULL;
		}
		double *daBuf = (double *)malloc(byteLength);
		*nbEle = byteLength/8;
		
		ldouble buf;
		for(i = 0;i<*nbEle;i++)
		{
			j = i*8;
			memcpy(buf.byte, bytes+j, 8);
			symTransform_8bytes(buf.byte);
			daBuf[i] = buf.value;
		}
		free(bytes);
		return daBuf;
	}
}

float *scidx_readFloatData(char *srcFilePath, size_t *nbEle, int *status)
{
	int state = scidx_SCES;
	if(scidx_dataEndianType==scidx_sysEndianType)
	{
		float *daBuf = scidx_readFloatData_systemEndian(srcFilePath, nbEle, &state);
		*status = state;
		return daBuf;
	}
	else
	{
		size_t i,j;
		
		size_t byteLength;
		unsigned char* bytes = scidx_readByteData(srcFilePath, &byteLength, &state);
		if(state == scidx_FERR)
		{
			*status = scidx_FERR;
			return NULL;
		}
		float *daBuf = (float *)malloc(byteLength);
		*nbEle = byteLength/4;
		
		lfloat buf;
		for(i = 0;i<*nbEle;i++)
		{
			j = i*4;
			memcpy(buf.byte, bytes+j, 4);
			symTransform_4bytes(buf.byte);
			daBuf[i] = buf.value;
		}
		free(bytes);
		return daBuf;
	}
}

double *scidx_readDoubleData_systemEndian(char *srcFilePath, size_t *nbEle, int *status)
{
	size_t inSize;
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 1\n");
        *status = scidx_FERR;
        return NULL;
    }
	fseek(pFile, 0, SEEK_END);
    inSize = ftell(pFile);
    *nbEle = inSize/8; //only support double in this version
    fclose(pFile);
    
    double *daBuf = (double *)malloc(inSize);
    
    pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 2\n");
        *status = scidx_FERR;
        return NULL;
    }
    fread(daBuf, 8, *nbEle, pFile);
    fclose(pFile);
    *status = scidx_SCES;
    return daBuf;
}

float *scidx_readFloatData_systemEndian(char *srcFilePath, size_t *nbEle, int *status)
{
	size_t inSize;
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 1\n");
        *status = scidx_FERR;
        return NULL;
    }
	fseek(pFile, 0, SEEK_END);
    inSize = ftell(pFile);
    *nbEle = inSize/4; 
    fclose(pFile);
    
    if(inSize<=0)
    {
		printf("Error: input file is wrong!\n");
		*status = scidx_FERR;
	}
    
    float *daBuf = (float *)malloc(inSize);
    
    pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 2\n");
        *status = scidx_FERR;
        return NULL;
    }
    fread(daBuf, 4, *nbEle, pFile);
    fclose(pFile);
    *status = scidx_SCES;
    return daBuf;
}

void scidx_writeByteData(unsigned char *bytes, size_t byteLength, char *tgtFilePath, int *status)
{
	FILE *pFile = fopen(tgtFilePath, "wb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 3\n");
        *status = scidx_FERR;
        return;
    }
    
    fwrite(bytes, 1, byteLength, pFile); //write outSize bytes
    fclose(pFile);
    *status = scidx_SCES;
}

void scidx_writeDoubleData(double *data, size_t nbEle, char *tgtFilePath, int *status)
{
	size_t i = 0;
	char s[64];
	FILE *pFile = fopen(tgtFilePath, "wb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 3\n");
        *status = scidx_FERR;
        return;
    }
    
    for(i = 0;i<nbEle;i++)
	{
		sprintf(s,"%.20G\n",data[i]);
		fputs(s, pFile);
	}
    
    fclose(pFile);
    *status = scidx_SCES;
}

void scidx_writeFloatData(float *data, size_t nbEle, char *tgtFilePath, int *status)
{
	size_t i = 0;
	char s[64];
	FILE *pFile = fopen(tgtFilePath, "wb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 3\n");
        *status = scidx_FERR;
        return;
    }
   
    for(i = 0;i<nbEle;i++)
	{
		//printf("i=%d\n",i);
		//printf("data[i]=%f\n",data[i]);
		sprintf(s,"%.30G\n",data[i]);
		fputs(s, pFile);
	}
    
    fclose(pFile);
    *status = scidx_SCES;
}


void scidx_writeIntData(int *data, size_t nbEle, char *tgtFilePath, int *status)
{
	size_t i = 0;
	char s[64];
	FILE *pFile = fopen(tgtFilePath, "wb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 3\n");
        *status = scidx_FERR;
        return;
    }
   
    for(i = 0;i<nbEle;i++)
	{
		//printf("i=%d\n",i);
		//printf("data[i]=%f\n",data[i]);
		snprintf(s, sizeof(s), "%d\n", data[i]);
		fputs(s, pFile);
	}
    
    fclose(pFile);
    *status = scidx_SCES;
}

void scidx_writeData(void *data, int dataType, size_t nbEle, char *tgtFilePath, int *status)
{
	int state = scidx_SCES;
	if(dataType == scidx_FLOAT)
	{
		float* dataArray = (float *)data;
		scidx_writeFloatData(dataArray, nbEle, tgtFilePath, &state);
	}
	else if(dataType == scidx_DOUBLE)
	{
		double* dataArray = (double *)data;
		scidx_writeDoubleData(dataArray, nbEle, tgtFilePath, &state);	
	}
	else
	{
		printf("Error: data type cannot be the types other than scidx_FLOAT or scidx_DOUBLE\n");
		*status = scidx_TERR; //wrong type
		return;
	}
	*status = state;
}

void scidx_writeFloatData_inBytes(float *data, size_t nbEle, char* tgtFilePath, int *status)
{
	size_t i = 0; 
	int state = scidx_SCES;
	lfloat buf;
	unsigned char* bytes = (unsigned char*)malloc(nbEle*sizeof(float));
	for(i=0;i<nbEle;i++)
	{
		buf.value = data[i];
		bytes[i*4+0] = buf.byte[0];
		bytes[i*4+1] = buf.byte[1];
		bytes[i*4+2] = buf.byte[2];
		bytes[i*4+3] = buf.byte[3];					
	}

	size_t byteLength = nbEle*sizeof(float);
	scidx_writeByteData(bytes, byteLength, tgtFilePath, &state);
	free(bytes);
	*status = state;
}

void scidx_writeDoubleData_inBytes(double *data, size_t nbEle, char* tgtFilePath, int *status)
{
	size_t i = 0, index = 0; 
	int state = scidx_SCES;
	ldouble buf;
	unsigned char* bytes = (unsigned char*)malloc(nbEle*sizeof(double));
	for(i=0;i<nbEle;i++)
	{
		index = i*8;
		buf.value = data[i];
		bytes[index+0] = buf.byte[0];
		bytes[index+1] = buf.byte[1];
		bytes[index+2] = buf.byte[2];
		bytes[index+3] = buf.byte[3];
		bytes[index+4] = buf.byte[4];
		bytes[index+5] = buf.byte[5];
		bytes[index+6] = buf.byte[6];
		bytes[index+7] = buf.byte[7];
	}

	size_t byteLength = nbEle*sizeof(double);
	scidx_writeByteData(bytes, byteLength, tgtFilePath, &state);
	free(bytes);
	*status = state;
}
