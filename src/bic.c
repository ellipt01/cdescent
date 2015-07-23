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
	if (cd->use_intercept && fabs (cd->b0) > 0.) mm_real_xj_add_const (r, 0, - cd->b0);	// r = y - mu - b
	rss = mm_real_xj_ssq (r, 0);	// rss = | y - mu |^2
	mm_real_free (r);
	// if not lasso, rss = | y - mu |^2 + lambda2 * | nu |^2
	// if (!cd->is_regtype_lasso && cd->nu) rss += cd->lambda2 * mm_real_xj_ssq (cd->nu, 0);
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
				double	gj = cd->lambda2 * cd->lreg->dtd[j];
				if (!cd->lreg->xnormalized) gj /= cd->lreg->xtx[j];
				df += 1. / (1. + gj);
			}
		}
	}
	return df;
}

/*** default function which evaluates BIC as: BIC = log (rss) + log (m) * df / m ***/
double
cdescent_default_bic_eval_func (const cdescent *cd, bic_info *info, void *data)
{
	double	val = log (info->rss) + info->df * log (info->m) / info->m;
	return val;
}

static bic_func *
bic_function_alloc (void)
{
	bic_func	*func = (bic_func *) malloc (sizeof (bic_func));
	func->function = NULL;
	func->data = NULL;
	return func;
}

/*** create bic_function object ***/
bic_func *
bic_function_new (const bic_eval_func function, void *data)
{
	bic_func	*func = bic_function_alloc ();
	func->function = function;
	func->data = data;
	return func;
}

/*** Bayesian Information Criterion
 * BIC = log(rss) + df * log(m) / m
 * rss		: residual sum of squares |b - Z * beta|^2
 * df		: degree of freedom
 * m		: number of data (number of rows of b and Z)
 * n		: number of variables (number of columns of Z and number of rows of beta) ***/
bic_info *
calc_bic_info (const cdescent *cd)
{
	bic_func	*bicfunc = cd->path->bicfunc;
	bic_info	*info;
	info = bic_info_alloc ();
	info->rss = calc_rss (cd);
	info->df = calc_degree_of_freedom (cd);
	info->m = (double) *cd->m;
	info->n = (double) *cd->n;
	info->bic_val = bicfunc->function (cd, info, bicfunc->data);
	return info;
}

