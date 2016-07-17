/*
 * elasticnet.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cdescent.h>

#include "example.h"

/*** default settings ***/
char		infn_x[80] = "\0";		// store input file name of design matrix
char		infn_y[80] = "\0";		// store input file name of observed data
double		alpha = 1.;				// raito of L1 / L2 penalty parameter
bool		constraint = false;
bool		use_fixed_lambda2 = false;
double		lambda2 = 0.;
double		log10_lambda = -2.;		// lower bound of log10(lambda1) for warm start
double		dlog10_lambda = 0.1;	// increment of log10(lambda1)

double		tolerance = 1.e-3;
int			maxiter = 100000;
bool		verbos = false;

double		lower = 0.;

/* constraint function */
bool
constraint_func0 (cdescent *cd, const int j, const double etaj, double *forced)
{
	*forced = lower;
	return (cd->beta->data[j] + etaj >= *forced);
}

/***************************************************
 * An example program of elastic net regression
 *           using cdescent library.
 ***************************************************/

int
main (int argc, char **argv)
{
	mm_dense	*x;
	mm_dense	*y;
	mm_real		*d;

	linregmodel	*lreg;

	cdescent	*cd;

	FILE		*fp;

	/*** read command line options ***/
	if (!read_params (argc, argv)) usage (argv[0]);

	/*** prepare observation, predictors and L2 penalty ***/
	/* read observation mm_dense *y from file */
	if ((fp = fopen (infn_y, "r")) == NULL) {
		fprintf (stderr, "ERROR: cannot open file %s.\n", infn_y);
		exit (1);
	}
	y = mm_real_fread (fp);
	fclose (fp);

	/* read predictors mm_real *x from file */
	if ((fp = fopen (infn_x, "r")) == NULL) {
		fprintf (stderr, "ERROR: cannot open file %s.\n", infn_x);
		exit (1);
	}
	x = mm_real_fread (fp);
	fclose (fp);

	/* L2 penalty */
	//	d = NULL;	// no L2 penalty for lasso

	// sparse identity matrix for elastic net
	d = mm_real_eye (MM_REAL_SPARSE, x->n);

	// sparse 1D derivation operator for s-lasso
	//	d = penalty_smooth (MM_REAL_SPARSE, x->n);	// see example.c

	standardizing (x, y);

	/*** create linear regression model object
	     for || y - x * beta ||^2 + lambda2 * || d * beta ||^2 ***/
//	lreg = linregmodel_new (y, x, d, DO_CENTERING_Y | DO_STANDARDIZING_X);
	lreg = linregmodel_new (y, x, d, DO_CENTERING_Y | DO_CENTERING_X);

	/*** create coordinate descent object ***/
	cd = cdescent_new (alpha, lreg, tolerance, maxiter, false);

	/*** set parameters of pathwise coordinate descent optimization ***/
	cdescent_set_log10_lambda_lower (cd, log10_lambda);
	cdescent_set_dlog10_lambda (cd, dlog10_lambda);
	if (constraint) cdescent_set_constraint (cd, constraint_func0);
	cdescent_set_outputs_fullpath (cd, NULL);	// output full solution path
	cdescent_set_outputs_info (cd, NULL);		// output regression info
	if (use_fixed_lambda2) cdescent_use_fixed_lambda2 (cd, lambda2);

	/*** do pathwise coordinate descent regression ***/
	cdescent_do_pathwise_optimization (cd);

	cdescent_free (cd);
	linregmodel_free (lreg);
	mm_real_free (y);
	mm_real_free (x);
	if (d) mm_real_free (d);

	return EXIT_SUCCESS;
}
