/**
 *  @file scidx_utility.h
 *  @author Sheng Di
 *  @date Feb, 2022
 *  @brief Header file for the utility.c.
 *  (C) 2023 by 
 * */

#ifndef _SCIDX_UTILITY_H
#define _SCIDX_UTILITY_H

extern struct timeval scidx_costStart; /*only used for recording the cost*/
extern double scidx_totalCost;

void scidx_cost_start();
void scidx_cost_end();
void scidx_cost_end_msg(char *);

#endif /* ----- #ifndef _SCIDX_UTILITY_H  ----- */
