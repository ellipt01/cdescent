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

		fprintf (fp, "%d\t%.4e\t%.4e\n", iter, cd->nrm1, cd->beta[i]);
		fclose (fp);
	}
	return;
}

void
example_cdescent_pathwise (const linreg *lreg, double tmin, double dt, double tmax, double tol, int maxiter)
{
	int			iter = 0;
	cdescent	*cd;

	/* warm start */
	cd = cdescent_alloc (lreg, tmax, tol);

	while (tmin <= cd->lambda1) {

		fprintf (stdout, "t = %f\n", cd->lambda1);
		if (!cdescent_cyclic (cd, maxiter)) break;
		output_solutionpath_cdescent (iter, cd);

		cd->lambda1 -= dt;
		iter++;
	}

	return;
}
