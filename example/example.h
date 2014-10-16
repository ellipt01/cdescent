/*
 * example.h
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#ifndef EXAMPLE_H_
#define EXAMPLE_H_

#include <cdescent.h>

char		infn_x[80];
char		infn_y[80];

/* example.c */
mm_real	*mm_real_penalty_smooth (MMRealFormat format, const int n);

/* example_cdescent.c */
void		example_cdescent_pathwise (cdescent *cd, double log10_lambda1_lower, double dlog10_lambda1, int maxiter, bool output_path, bool output_bic);

#endif /* EXAMPLE_H_ */
