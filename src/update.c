/*
 * cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <math.h>
#include <cdescent.h>

#include "private.h"

/*** updater of intercept: (sum (y) - sum(X) * beta) / n ***/
void
cdescent_update_intercept (cdescent *cd)
{
	if (cd->lreg->ycentered && cd->lreg->xcentered) cd->b = 0.;
	else {
		double	nb = 0.;	// n * b
		// b += bar(y)
		if (!cd->lreg->ycentered) nb += cd->lreg->sy;
		// b -= bar(X) * beta
		if (!cd->lreg->xcentered) nb -= ddot_ (&cd->lreg->x->n, cd->lreg->sx, &ione, cd->beta->data, &ione);
		cd->b = nb / (double) cd->lreg->x->m;
	}
	return;
}

void
cdescent_update_beta (cdescent *cd, const int j, const double etaj)
{
	// beta[j] += eta[j]
	cd->beta->data[j] += etaj;
	return;
}

void
cdescent_update_mu (cdescent *cd, const int j, const double etaj)
{
	// mu += etaj * X(:,j)
	mm_real_axjpy (etaj, j, cd->lreg->x, cd->mu);
	return;
}

void
cdescent_update_nu (cdescent *cd, const int j, const double etaj)
{
	// nu += etaj * D(:,j)
	if (linregmodel_is_regtype_lasso (cd->lreg)) return;
	mm_real_axjpy (etaj, j, cd->lreg->d, cd->nu);
	return;
}

/*** progress coordinate descent update for one full cycle ***/
bool
cdescent_update_cyclic_once_cycle (cdescent *cd)
{
	int		j;
	double	nrm2;

	/* b = (sum(y) - sum(X) * beta) / n.
	 * so, if y or X are not centered,
	 * i.e. sum(y) != 0 or sum(X) != 0,
	 * b must be updated on each cycle. */
	if (!cd->lreg->xcentered) cdescent_update_intercept (cd);

	nrm2 = 0.;
	/*** single "one-at-a-time" update of cyclic coordinate descent ***/
	for (j = 0; j < cd->lreg->x->n; j++) {
		// era[j] = beta[j] - beta_prev[j]
		double	etaj = cdescent_beta_stepsize (cd, j);

		if (fabs (etaj) > 0.) {
			cdescent_update_beta (cd, j, etaj);
			cdescent_update_mu (cd, j, etaj);
			cdescent_update_nu (cd, j, etaj);
			nrm2 += pow (etaj, 2.);
		}

	}
	cd->nrm1 = mm_real_asum (cd->beta);

	return (sqrt (nrm2) < cd->tolerance);
}

/*
bool
cdescent_update_cyclic_once_cycle (cdescent *cd)
{
	int		n = 3;
	int		j;
	double	nrm2;

	cdescent_update_intercept (cd);

	nrm2 = 0.;

	for (j = 0; j < cd->lreg->x->n - n; j += n) {
		int		k;
		double	eta[n];
		// do parallel
		for (k = 0; k < n; k++) eta[k] = cdescent_beta_stepsize (cd, j + k);
		for (k = 0; k < n; k++) {
			if (fabs (eta[k]) > 0.) {
				cdescent_update_beta (cd, j + k, eta[k]);
				cdescent_update_mu (cd, j + k, eta[k]);
				cdescent_update_nu (cd, j + k, eta[k]);
				nrm2 += pow (eta[k], 2.);
			}
		}
	}
	cd->nrm1 = mm_real_asum (cd->beta);

	return (sqrt (nrm2) < cd->tolerance);
}
*/

/*** cyclic coordinate descent ***/
/* repeat coordinate descent until solution is converged */
bool
cdescent_update_cyclic (cdescent *cd, const int maxiter)
{
	int		iter = 0;
	bool	converged = false;

	while (!converged) {

		converged = cdescent_update_cyclic_once_cycle (cd);

		if (++iter >= maxiter) {
			cdescent_warning ("cdescent_cyclic", "reaching max number of iterations.", __FILE__, __LINE__);
			break;
		}
	}

	return converged;
}
