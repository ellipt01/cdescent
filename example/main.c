/*
 * main.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "example.h"

int			skipheaders = 0;
double		lambda2 = 0.;
double		start = -2.;
double		stop = 10.;
double		dt = 0.1;
double		gamma_bic = 0.;	// traditional BIC
int			maxiter = 100000;

int
main (int argc, char **argv)
{
	mm_dense		*x;
	mm_dense		*y;
	mm_real		*d;

	linregmodel	*lreg;

	if (!read_params (argc, argv)) usage (argv[0]);
	fprintf_params (stderr);

	/* linear system */
	{
		int		m;
		int		n;
		double	*datax;
		double	*datay;
		read_data (infn, skipheaders, &m, &n, &datay, &datax);
		y = create_mm_dense (m, 1, datay);
		free (datay);
		x = create_mm_dense (m, n, datax);
		free (datax);
	}
	//	d = NULL;
	d = mm_real_eye (MM_REAL_SPARSE, x->n);
	//	d = mm_real_penalty_smooth (MM_REAL_SPARSE, x->n);

	lreg = linregmodel_new (y, x, lambda2, d, false, true, true, true);

	{
#ifdef _OPENMP
		double	t1, t2;
		t1 = omp_get_wtime ();
#endif
		example_cdescent_pathwise (lreg, start, dt, stop, 1.e-3, maxiter, true);
#ifdef _OPENMP
		t2 = omp_get_wtime ();
		fprintf (stderr, "time = %.2e\n", t2 - t1);
#endif
	}

	linregmodel_free (lreg);
	mm_real_free (x);
	mm_real_free (y);
	if (d) mm_real_free (d);

	return EXIT_SUCCESS;
}
