/*
 * bic.c
 *
 *  Created on: 2014/07/15
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "private/private.h"

/*c****************************************************************************
 *c   Bayesian Information Criterion for linear regression model y = X * beta
 *c****************************************************************************/

/* allocate bic_info */
static bic_info *
bic_info_alloc (void)
{
	bic_info	*info = (bic_info *) malloc (sizeof (bic_info));
	info->m = 0.;
	info->n = 0.;
	info->rss = 0.;
	info->df = 0.;
	info->gamma = 0.;
	info->bic_val = CDESCENT_POSINF;
	return info;
}

/* residual sum of squares
 * rss = | y - mu - b |^2 */
static double
calc_rss (const cdescent *cd)
{
	double	rss;
	mm_dense	*r = mm_real_copy (cd->lreg->y);	// r = y
	mm_real_axjpy (-1., cd->mu, 0, r);	// r = y - mu
	// intercept
	if (fabs (cd->b) > 0.) mm_real_xj_add_const (r, 0, - cd->b);	// r = y - mu - b
	rss = mm_real_xj_ssq (r, 0);	// rss = | y - mu |^2
	mm_real_free (r);
	// if not lasso, rss = | y - mu |^2 + lambda2 * | nu |^2
//	if (!cd->is_regtype_lasso && cd->nu) rss += cd->lreg->lambda2 * mm_real_xj_ssq (cd->nu, 0);
	return rss;
}

/* degree of freedom
 * A = {j | beta_j != 0} : active set
 * df = trace( X(A) * ( X(A)'*X(A) + lambda2*D(A)'*D(A) ) * X(A)' )
 * Under the orthogonal covariance matrix assumption (i.e., X'*X = I),
 * and assuming lambda2*D'*D is diagonal,
 * df = sum_{j in A} 1 / (1 + lambda2 * D(:,j)'*D(:,j) / X(:,j)'*X(:,j))
 * (Hebiri, 2008) */
static double
calc_degree_of_freedom (const cdescent *cd)
{
	int		j;
	double	df = 0.;
	for (j = 0; j < cd->beta->nnz; j++) {
		if (fabs (cd->beta->data[j]) > 0.) {
			if (cd->is_regtype_lasso) df += 1.;
			else {
				double	gj = cd->lreg->lambda2 * cd->lreg->dtd[j];
				if (!cd->lreg->xnormalized) gj /= cd->lreg->xtx[j];
				df += 1. / (1. + gj);
			}
		}
	}
	return df;
}

/*** Extended Bayesian Information Criterion (Chen and Chen, 2008)
 * eBIC = log(rss) + df * ( log(m) + 2 * gamma * log(n) ) / m
 * gamma	: tuning parameter for eBIC
 * rss		: residual sum of squares |b - Z * beta|^2
 * df		: degree of freedom
 * m		: number of data (number of rows of b and Z)
 * n		: number of variables (number of columns of Z and number of rows of beta)
 * if gamma = 0, eBIC is identical with the classical BIC ***/
bic_info *
cdescent_eval_bic (const cdescent *cd, const double gamma)
{
	bic_info	*info;
	if (gamma < 0.) {
		printf_warning ("cdescent_eval_bic", "gamma must be >= 0.", __FILE__, __LINE__);
		return NULL;
	}
	info = bic_info_alloc ();
	info->gamma = gamma;
	info->rss = calc_rss (cd);
	info->df = calc_degree_of_freedom (cd);
	info->m = (double) *cd->m;
	info->n = (double) *cd->n;
	info->bic_val = log (info->rss) + info->df * log (info->m) / info->m;
	if (fabs (gamma) > 0.) info->bic_val += 2. * info->df * info->gamma * log (info->n) / info->m;
	return info;
}

