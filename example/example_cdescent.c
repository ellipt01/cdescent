/*
 * example_cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <cdescent.h>

static void
output_solutionpath_cdescent (int iter, const cdescent *cd)
{
	int			i;
	char		fn[80];
	FILE		*fp;
	int			p = cd->lreg->p;
	double		*beta = cdescent_copy_beta (cd, true);
	double		a = (cd->lreg->pen) ? cd->lreg->pen->a : 1.;
	double		nrm1 = cd->nrm1 * (1. + a * cd->lreg->lambda2);	// cd->nrm1 / cd->lreg->scale2;

	for (i = 0; i < p; i++) {

		sprintf (fn, "beta%03d.res", i);

		if (iter == 0) fp = fopen (fn, "w");
		else fp = fopen (fn, "aw");
		if (fp == NULL) continue;

		fprintf (fp, "%d\t%.4e\t%.4e\n", iter, nrm1, beta[i]);
		fclose (fp);
	}
	free (beta);
	return;
}

void
example_cdescent_pathwise (const linreg *lreg, double tmin, double dt, double tmax, double tol, int maxiter)
{
	int			iter = 0;
	double		t;
	cdescent	*cd = cdescent_alloc (lreg, t, tol);

	/* warm start */
	t = tmax;
	//	t = (tmax < cd->camax) ? tmax : cd->camax;
	fprintf (stderr, "tmax = %.2f\n", t);
	while (tmin <= t) {

		cdescent_set_lambda1 (cd, t);
		if (!cdescent_cyclic (cd, maxiter)) break;
		output_solutionpath_cdescent (iter, cd);

		t -= dt;
		fprintf (stdout, "t = %f\n", t);
		iter++;
	}

	return;
}
