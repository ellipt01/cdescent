/*
 * elasticnet_reweight.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cdescent.h>

#include "example.h"
#include "settings.h"

/*********************************************************
 * An example program of reweighted elastic net regression
 *             using cdescent library.
 *********************************************************/

/* anternative of L0 norm regression after Candes et al.,2008 */
mm_dense *
weight_func0 (cdescent *cd, void *data)
{
	int			i;
	int			m = cd->beta->m;
	mm_dense	*w = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, 1, m);
	double		eps = *(double *) data;
	for (i = 0; i < m; i++) w->data[i] = 1. / (fabs (cd->beta->data[i]) + eps);
	return w;
}

int
main (int argc, char **argv)
{
	mm_dense	*x;
	mm_dense	*y;
	mm_real		*d;

	linregmodel	*lreg;

	cdescent	*cd;

	double				eps = 1.e-3;
	reweighting_func	*func;

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
	cdescent_set_pathwise_outputs_fullpath (cd, NULL);	// output full solution path
	cdescent_set_pathwise_outputs_bic_info (cd, NULL);	// output BIC info
	cdescent_set_pathwise_gamma_bic (cd, gamma_bic);	// set gamma for eBIC

	/* setup reweighting */
	func  = reweighting_function_new (1., weight_func0, &eps);
	cdescent_set_reweighting (cd, maxiter, tolerance, func);

	/*** do reweighted pathwise coordinate descent regression ***/
	cdescent_do_pathwise_optimization (cd);
	fprintf (stderr, "lambda_opt = %.2f, nrm1(beta_opt) = %.2f, min_bic = %.2f\n",
		cd->path->lambda_opt, cd->path->nrm1_opt, cd->path->min_bic_val);

	fprintf (stderr, "lambda1_rwt = %.2f, nrm1(beta_rwt) = %.2f\n", cd->lambda1, cd->nrm1);

	cdescent_free (cd);
	linregmodel_free (lreg);

	mm_real_free (y);
	mm_real_free (x);
	if (d) mm_real_free (d);

	return EXIT_SUCCESS;
}
