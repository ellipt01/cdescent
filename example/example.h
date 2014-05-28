/*
 * example.h
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#ifndef EXAMPLE_H_
#define EXAMPLE_H_

/* example.c */
void	read_data (char *fn, int skip_header, size_t *n, size_t *p, double **y, double **x);

/* example_larsen.c */
void	example_larsen (const linreg *lreg, double start, double dt, double stop, double gamma, int maxiter);

/* example_cdescent.c */
void	example_cdescent_cyclic (const linreg *lreg, double start, double dt, double stop, double tol, int maxiter);

#endif /* EXAMPLE_H_ */
