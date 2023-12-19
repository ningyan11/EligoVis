/**
 *  @file scidx_utility.c
 *  @author Sheng Di
 *  @date Dec, 2023
 *  @brief 
 *  (C) 2023 by
 */


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "scidx_utility.h"
#include "scidx_defines.h"

struct timeval scidx_costStart; /*only used for recording the cost*/
double scidx_totalCost = 0;

void scidx_cost_start()
{
	scidx_totalCost = 0;
	gettimeofday(&scidx_costStart, NULL);
}

void scidx_cost_end()
{
	double elapsed;
	struct timeval costEnd;
	gettimeofday(&costEnd, NULL);
	elapsed = ((costEnd.tv_sec*1000000+costEnd.tv_usec)-(scidx_costStart.tv_sec*1000000+scidx_costStart.tv_usec))/1000000.0;
	scidx_totalCost += elapsed;
}

void scidx_cost_end_msg(char *msg)
{
    double elapsed;
    struct timeval costEnd;
    gettimeofday(&costEnd, NULL);
    elapsed = ((costEnd.tv_sec*1000000+costEnd.tv_usec)-(scidx_costStart.tv_sec*1000000+scidx_costStart.tv_usec))/1000000.0;
    scidx_totalCost += elapsed;
    printf("timecost=%f, %s\n", elapsed, msg);
}
