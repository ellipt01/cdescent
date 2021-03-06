/*
 * example.h
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#ifndef EXAMPLE_H_
#define EXAMPLE_H_

#include <stdbool.h>

void	standardizing (mm_real *x, mm_real *y);
mm_real	*penalty_smooth (MMRealFormat format, const int n);
void	usage (char *toolname);
bool	read_params (int argc, char **argv);

#endif /* EXAMPLE_H_ */
