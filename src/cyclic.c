/*
 * cyclic.c
 *
 *  Created on: 2015/07/23
 *      Author: utsugi
 */

#include <stdlib.h>
#include <cdescent.h>
#include <mmreal.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* update.c */
extern void		update_intercept (cdescent *cd);
extern void		cdescent_update (cdescent *cd, int j, double *amax_eta);
extern void		cdescent_update_atomic (cdescent *cd, int j, double *amax_eta);

/*** progress cyclic coordinate descent update for one full cycle ***/
bool
cdescent_do_update_once_cycle_cyclic (cdescent *cd)
{
	int		j;
	int		n = *cd->n;
	double	amax_eta;	// max of |eta(j)| = |beta_new(j) - beta_prev(j)|

	/* b = (sum(y) - sum(X) * beta) / m */
	if (cd->use_intercept && cd->lreg->xcentered) update_intercept (cd);

	amax_eta = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent
	 * the following code was referring to shotgun by A.Kyrola,
	 * https://github.com/akyrola/shotgun ****/
	if (cd->parallel) {
#pragma omp parallel for
		for (j = 0; j < n; j++) cdescent_update_atomic (cd, j, &amax_eta);
	} else {
		for (j = 0; j < n; j++) cdescent_update (cd, j, &amax_eta);
	}

	cd->nrm1 = mm_real_xj_asum (cd->beta, 0);

	if (!cd->was_modified) cd->was_modified = true;

	return (amax_eta < cd->tolerance);
}
