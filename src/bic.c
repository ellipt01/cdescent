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

/*c*******************************************************
 *c   Bayesian Information Criterion for L2 regularized
 *c   linear regression model b = Z * beta
 *c   where b = [y ; 0], Z = [x ; sqrt(lambda2) * D]
 *c*******************************************************/

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
	info->bic_val = CDESCENT_POS_INF;
	return info;
}

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
	// intercept
	if (fabs (cd->b) > 0.) {
		int		j;
		for (j = 0; j < cd->mu->nz; j++) r[j] -= cd->b;	// r = y - mu - b
	}
	rss = ddot_ (&cd->lreg->y->nz, r, &ione, r, &ione);	// rss = | y - mu |^2
	free (r);
	return rss;
}

/* degree of freedom
 * A = {j ; beta_j != 0}
 * df = trace( X(A) * ( X(A)'*X(A) + lambda2*D(A)'*D(A) ) * X(A)' )
 * Under the orthogonal covariance matrix assumption (i.e., X(A)'*X(A) = I(A)),
 * df -> sum_{j in A} 1 / (1 + lambda2 * D(:,j)' * D(:,j)) */
static double
calc_degree_of_freedom (const cdescent *cd)
{
	int		j;
	double	df = 0.;
	for (j = 0; j < cd->beta->nz; j++) {
		if (fabs (cd->beta->data[j]) > 0.) {
			if (cd->lreg->is_regtype_lasso) df += 1.;
			else df += 1. / (1. + cd->lreg->lambda2 * cd->lreg->dtd[j]);
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
	info->m = (double) cd->lreg->x->m;
	info->n = (double) cd->lreg->x->n;
	info->bic_val = log (info->rss) + info->df * log (info->m) / info->m;
	if (fabs (gamma) > 0.) info->bic_val += 2. * info->df * info->gamma * log (info->n) / info->m;
	return info;
}

