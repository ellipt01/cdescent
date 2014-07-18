/*
 * example_cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "example.h"

#ifdef DEBUG
extern double	gamma_bic;
#endif

static void
cdescent_output_solutionpath (int iter, const cdescent *cd)
{
	int			i;
	char		fn[80];
	FILE		*fp;

	for (i = 0; i < cd->lreg->x->n; i++) {

		sprintf (fn, "beta%03d.res", i);

		if (iter == 0) fp = fopen (fn, "w");
		else fp = fopen (fn, "aw");
		if (fp == NULL) continue;

		fprintf (fp, "%d\t%.4e\t%.4e\n", iter, cd->nrm1, cd->beta->data[i]);
		fclose (fp);
	}
	return;
}

/* Evaluate the regression coefficients beta
 * correspond to the specified L1 regularization parameter log_lambda1.
 *
 * The evaluation is made starting at the smallest value λmax for which
 * the entire vector β = 0, and decreasing sequence of values for λ1 on
 * the log scale.
 * log (λmax) is identical with log ( max ( abs(X' * y) ) ), where this
 * value is stored in cd->lreg->logcamax.
 * The interval of decreasing sequence of log (λ1) is specified by
 * dlog_lambda1.
 *
 * if output_path == true, solution path is output in files beta0xx.res
 */

void
example_cdescent_pathwise (cdescent *cd, double log10_lambda1, double dlog10_lambda1, int maxiter, bool output_path)
{
	int			iter = 0;
	double		logt;

	/* warm start */
	logt = cd->lreg->logcamax;

	while (log10_lambda1 <= logt) {

		cdescent_set_log10_lambda1 (cd, logt);

		if (!cdescent_update_cyclic (cd, maxiter)) break;

		// output solution path
		if (output_path) cdescent_output_solutionpath (iter++, cd);

#ifdef DEBUG
		double	bic = cdescent_eval_bic (cd, gamma_bic);
		fprintf (stdout, "t %.4e ebic %.8e\n", cd->nrm1, bic);
#endif

		logt -= dlog10_lambda1;
	}
	fprintf (stderr, "total iter = %d\n", cd->total_iter);
	return;
}
