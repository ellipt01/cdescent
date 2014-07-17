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
output_solutionpath_cdescent (int iter, const cdescent *cd)
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

void
example_cdescent_pathwise (const linregmodel *lreg, double logtmin, double dlogt, double logtmax, double tol, int maxiter, bool parallel)
{
	int			iter = 0;
	double		logt;
	cdescent	*cd;

	/* warm start */
	cd = cdescent_new (lreg, tol, parallel);
	logt = (cd->lreg->logcamax <= logtmax) ? cd->lreg->logcamax : logtmax;

	while (logtmin <= logt) {

		cdescent_set_log10_lambda1 (cd, logt);

		if (!cdescent_update_cyclic (cd, maxiter)) break;
		output_solutionpath_cdescent (iter, cd);
#ifdef DEBUG
		double	bic = cdescent_eval_bic (cd, gamma_bic);
		fprintf (stdout, "t %.4e ebic %.8e\n", cd->nrm1, bic);
#endif

		logt -= dlogt;

		iter++;
	}
	fprintf (stderr, "total iter = %d\n", cd->total_iter);
	cdescent_free (cd);
	return;
}
