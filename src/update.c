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

/* update intercept: (sum (y) - sum(X) * beta) / m
 * intercept is calculated in original scale */
void
update_intercept (cdescent *cd)
{
	cd->b0 = 0.;
	// b += bar(y)
	if (cd->lreg->ycentered && cd->lreg->sy) cd->b0 += *(cd->lreg->sy);
	// b -= bar(X) * beta
	if (cd->lreg->xcentered && cd->lreg->sx) {
		mm_dense	*beta = cdescent_get_beta_in_original_scale (cd);
		cd->b0 -= ddot_ (cd->n, cd->lreg->sx, &ione, beta->data, &ione);
		mm_real_free (beta);
	}
	cd->b0 /= (double) *cd->m;
	return;
}

static void
update_betaj (cdescent *cd, const int j, double *etaj, double *abs_etaj)
{
	double	val;
	// constraint coordinate descent after Franc, Hlavac and Navara, 2005.
	if (cd->cfunc && !cd->cfunc (cd, j, *etaj, &val)) {
		*etaj = - cd->beta->data[j] + val;
		cd->beta->data[j] = val;
	} else cd->beta->data[j] += *etaj;
	*abs_etaj = fabs (*etaj);
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
	update_betaj (cd, j, &etaj, &abs_etaj);
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
	update_betaj (cd, j, &etaj, &abs_etaj);
	// update mu (= X * beta): mu += etaj * X(:,j)
	mm_real_axjpy_atomic (etaj, cd->lreg->x, j, cd->mu);
	// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += etaj * D(:,j)
	if (!cd->is_regtype_lasso) mm_real_axjpy_atomic (etaj, cd->lreg->d, j, cd->nu);
	// update max( |etaj| )
	atomic_max (amax_eta, abs_etaj);

	return;
}
