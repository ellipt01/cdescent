/*
 * adaelasticnet.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>

#include <cdescent.h>

#include "example.h"
#include "settings.h"

double	dzero = 0.;

/* nonnegative constraint function */
bool
nonnegative_constraint (const double betaj, double *forced)
{
	*forced = dzero;
	return (betaj >= *forced);
}

/*********************************************************
 * An example program of adaptive elastic net regression
 *             using cdescent library.
 *********************************************************/

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

	/*** create linear regression model object
	     for || y - x * beta ||^2 + lambda2 * || d * beta ||^2 ***/
	lreg = linregmodel_new (y, x, d, DO_CENTERING_Y | DO_STANDARDIZING_X);

	/*** create coordinate descent object ***/
	cd = cdescent_new (alpha, lreg, tolerance, maxiter, false);

	/*** set parameters of pathwise coordinate descent optimization ***/
	cdescent_set_pathwise_log10_lambda_lower (cd, log10_lambda);
	cdescent_set_pathwise_dlog10_lambda (cd, dlog10_lambda);
	if (nonnegative) cdescent_set_constraint (cd, nonnegative_constraint);
	if (use_fixed_lambda2) cdescent_use_fixed_lambda2 (cd, lambda2);

	/*** do pathwise coordinate descent regression to obtain low bias solution ***/
	cdescent_do_pathwise_optimization (cd);

	/*** adaptive lasso ***/
	/* use low bias solution (beta of lambda = 10^log10_lambda_lower) for L1 / L2 norm weight */
	cdescent_set_penalty_factor (cd, cd->beta, 0.25);	// set weight = | beta_low_bias |^(1/4)

	cdescent_set_pathwise_outputs_fullpath (cd, NULL);	// output full solution path
	cdescent_set_pathwise_outputs_bic_info (cd, NULL);	// output BIC info
	/* do pathwise coordinate descent again */
	cdescent_do_pathwise_optimization (cd);

	fprintf (stderr, "lambda_opt = %.2f, nrm1(beta_opt) = %.2f, min_bic = %.2f\n",
		cd->path->lambda_opt, cd->path->nrm1_opt, cd->path->min_bic_val);

	cdescent_free (cd);
	linregmodel_free (lreg);

	mm_real_free (y);
	mm_real_free (x);
	if (d) mm_real_free (d);

	return EXIT_SUCCESS;
}

