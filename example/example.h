/*
 * example.h
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#ifndef EXAMPLE_H_
#define EXAMPLE_H_

#include <cdescent.h>

char		infn[80];

/* example.c */
void		usage (char *toolname);
bool		read_params (int argc, char **argv);
void		fprintf_params (FILE *stream);
void		read_data (char *fn, int skip_header, int *n, int *p, double **y, double **x);
mm_dense	*create_mm_dense (int m, int n, double *data);
mm_real	*mm_real_penalty_smooth (MMRealFormat format, const int n);

/* example_cdescent.c */
void		example_cdescent_pathwise (const linregmodel *lreg, double start, double dt, double stop, double tol,
				int maxiter, bool parallel);

#endif /* EXAMPLE_H_ */
