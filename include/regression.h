/*
 * regression.h
 *
 *  Created on: 2015/05/11
 *      Author: utsugi
 */

#ifndef REGRESSION_H_
#define REGRESSION_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/* regression.c */
bool		cdescent_do_update_one_cycle (cdescent *cd);
bool		cdescent_do_pathwise_optimization (cdescent *cd);

#ifdef __cplusplus
}
#endif

#endif /* REGRESSION_H_ */
