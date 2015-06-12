/*
 * elasticnet.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>

#include <cdescent.h>

#include "example.h"
#include "settings.h"

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

	/*** create linear regression model object
	     for || y - x * beta ||^2 + lambda2 * || d * beta ||^2 ***/
	lreg = linregmodel_new (y, x, lambda2, d, DO_CENTERING_Y | DO_STANDARDIZING_X);

	/*** create coordinate descent object ***/
	cd = cdescent_new (lreg, tolerance, maxiter, false);

	/*** set parameters of pathwise coordinate descent optimization ***/
	cdescent_set_pathwise_log10_lambda1_lower (cd, log10_lambda1);
	cdescent_set_pathwise_dlog10_lambda1 (cd, dlog10_lambda1);
	cdescent_set_pathwise_outputs_fullpath (cd, NULL);	// output full solution path
	cdescent_set_pathwise_outputs_bic_info (cd, NULL);	// output BIC info
	cdescent_set_pathwise_gamma_bic (cd, gamma_bic);	// set gamma for eBIC

	/*** do pathwise coordinate descent regression ***/
	cdescent_do_pathwise_optimization (cd);

	fprintf (stderr, "lambda1_opt = %.2f, nrm1(beta_opt) = %.2f, min_bic = %.2f\n",
		cd->path->lambda1_opt, cd->path->nrm1_opt, cd->path->min_bic_val);

	cdescent_free (cd);
	linregmodel_free (lreg);
	mm_real_free (y);
	mm_real_free (x);
	if (d) mm_real_free (d);

	return EXIT_SUCCESS;
}
