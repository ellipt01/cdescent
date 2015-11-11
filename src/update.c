/*
 * update.c
 *
 *  Created on: 2015/07/23
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>
#include <mmreal.h>

#include "private/private.h"
#include "private/atomic.h"

/* stepsize.c */
extern double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* update intercept: (sum (y) - sum(X) * beta) / n */
void
update_intercept (cdescent *cd)
{
	cd->b0 = 0.;
	// b += bar(y)
	if (!cd->lreg->ycentered) cd->b0 += *(cd->lreg->sy);
	// b -= bar(X) * beta
	if (!cd->lreg->xcentered) cd->b0 -= ddot_ (cd->n, cd->lreg->sx, &ione, cd->beta->data, &ione);
	if (fabs (cd->b0) > 0.) cd->b0 /= (double) *cd->m;
	return;
}

static void
update_betaj (double *betaj, double *etaj, double *abs_etaj, constraint_func func)
{
	// constraint coordinate descent after Franc, Hlavac and Navara, 2005.
	if (func) {
		double	val;
		double	betaj_val = *betaj;
		if (!func (betaj_val + *etaj, &val)) {
			*etaj = - betaj_val + val;
			*abs_etaj = fabs (*etaj);
			*betaj = val;
			return;
		}
	}
	*betaj += *etaj;
	return;
}

/* update beta, mu, nu and amax_eta */
void
cdescent_update (cdescent *cd, int j, double *amax_eta)
{
	// eta(j) = beta_new(j) - beta_prev(j)
	double	etaj = cdescent_beta_stepsize (cd, j);
	double	abs_etaj = fabs (etaj);

	if (abs_etaj < DBL_EPSILON) return;

	// update beta: beta(j) += eta(j)
	update_betaj (&cd->beta->data[j], &etaj, &abs_etaj, cd->cfunc);
	// update mu (= X * beta): mu += eta(j) * X(:,j)
	mm_real_axjpy (etaj, cd->lreg->x, j, cd->mu);
	// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += eta(j) * D(:,j)
	if (!cd->is_regtype_lasso) mm_real_axjpy (etaj, cd->lreg->d, j, cd->nu);
	// update max( |eta| )
	if (*amax_eta < abs_etaj) *amax_eta = abs_etaj;

	return;
}

/* update beta, mu, nu and amax_eta in atomic */
void
cdescent_update_atomic (cdescent *cd, int j, double *amax_eta)
{
	// eta(j) = beta_new(j) - beta_prev(j)
	double	etaj = cdescent_beta_stepsize (cd, j);
	double	abs_etaj = fabs (etaj);

	if (abs_etaj < DBL_EPSILON) return;

	// update beta: beta(j) += etaj
	update_betaj (&cd->beta->data[j], &etaj, &abs_etaj, cd->cfunc);
	// update mu (= X * beta): mu += etaj * X(:,j)
	mm_real_axjpy_atomic (etaj, cd->lreg->x, j, cd->mu);
	// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += etaj * D(:,j)
	if (!cd->is_regtype_lasso) mm_real_axjpy_atomic (etaj, cd->lreg->d, j, cd->nu);
	// update max( |etaj| )
	atomic_max (amax_eta, abs_etaj);

	return;
}
