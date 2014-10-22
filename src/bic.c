/*
 * bic.c
 *
 *  Created on: 2014/07/15
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "private.h"

/* cdescent.c */
extern double	cdescent_scale2 (const cdescent *cd, const int j);

static bic_info *
bic_info_alloc (void)
{
	return (bic_info *) malloc (sizeof (bic_info));
}

bic_info *
bic_info_new (void)
{
	bic_info	*info = bic_info_alloc ();
	info->m = 0.;
	info->n = 0.;
	info->rss = 0.;
	info->df = 0.;
	info->gamma = 0.;
	info->bic_val = + 1.0 / 0.0;	// positive inf
	return info;
}

void
bic_info_free (bic_info *info)
{
	if (info) free (info);
	return;
}

/*   Bayesian Information Criterion for L2 regularized
 *   linear regression model b = Z * beta
 *   where b = [y ; 0], Z = [x ; sqrt(lambda2) * D] */

/* residual sum of squares
 * rss = | b - Z * beta |^2
 *     = | y - mu |^2 + | 0 - sqrt(lambda2) * nu |^2 */
static double
calc_rss (const cdescent *cd)
{
	double	rss;
	double	*r = (double *) malloc (cd->lreg->y->nz * sizeof (double));
	dcopy_ (&cd->lreg->y->nz, cd->lreg->y->data, &ione, r, &ione);	// r = y
	daxpy_ (&cd->mu->nz, &dmone, cd->mu->data, &ione, r, &ione);	// r = y - mu
	rss = pow (dnrm2_ (&cd->lreg->y->nz, r, &ione), 2.);	// rss = | y - mu |^2
	free (r);
	// rss += | 0 - sqrt(lambda2) * nu |^2
	if (!cd->lreg->is_regtype_lasso)
		rss += cd->lreg->lambda2 * pow (dnrm2_ (&cd->nu->nz, cd->nu->data, &ione), 2.);
	return rss;
}

/* degree of freedom
 * A = {j ; beta_j != 0}
 * df = sum_{j in A} 1 (Efron et al., 2004)
 * df = sum_{j in A} 1 / (xtx[j] + lambda2 * dtd[j]) (Hebiri, 2008, Theorem 5.) */
static double
calc_degree_of_freedom (const cdescent *cd)
{
	int		j;
	double	df = 0.;
	for (j = 0; j < cd->beta->nz; j++) {
		if (fabs (cd->beta->data[j]) > 0.) {
			df += (cd->lreg->is_regtype_lasso) ? 1. : 1. / cdescent_scale2 (cd, j);
		}
	}
	return df;
}

/*** Extended Bayesian Information Criterion (Chen and Chen, 2008)
 * eBIC = log(rss) + df * ( log(m) + 2 * gamma * log(n) ) / m
 * gamma	: tuning parameter for eBIC
 * rss		: residual sum of squares |b - Z * beta|^2
 * df		: degree of freedom
 * m		: number of data (number rows of b and Z)
 * n		: number of variables (number columns of Z and number rows of beta)
 * if gamma = 0, eBIC is identical with the classical BIC ***/
bic_info *
cdescent_eval_bic (const cdescent *cd, const double gamma)
{
	bic_info	*info;
	if (gamma < 0.) error_and_exit ("cdescent_eval_bic", "gamma must be >= 0.", __FILE__, __LINE__);
	info = bic_info_new ();
	info->gamma = gamma;
	info->rss = calc_rss (cd);
	info->df = calc_degree_of_freedom (cd);
	info->m = (double) cd->lreg->x->m;
	info->n = (double) cd->lreg->x->n;
	if (!cd->lreg->is_regtype_lasso) info->m += (double) cd->lreg->d->m;
	info->bic_val = log (info->rss) + info->df * log (info->m) / info->m;
	if (fabs (gamma) > 0.) info->bic_val += 2. * info->df * info->gamma * log (info->n) / info->m;
	return info;
}

