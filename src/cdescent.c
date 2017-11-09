/*
 * cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cdescent.h>

#include "private/private.h"

// default file to output solution path
static const char	default_fn_path[] = "beta_path.data";

// default file to output regression info
static const char	default_fn_info[] = "regression_info.data";

double			log10_lambda_upper_default = 0.;

/*******************************
 *  coordinate descent object  *
 *******************************/

/* allocate cdescent object */
static cdescent *
cdescent_alloc (void)
{
	cdescent	*cd = (cdescent *) malloc (sizeof (cdescent));
	if (cd == NULL) return NULL;

	cd->was_modified = false;

	cd->is_regtype_lasso = true;
	cd->use_intercept = true;
	cd->use_fixed_lambda = false;
	cd->rule = CDESCENT_SELECTION_RULE_CYCLIC;

	cd->m = NULL;
	cd->n = NULL;
	cd->lreg = NULL;

	cd->alpha1 = 0.;
	cd->alpha2 = 0.;
	cd->lambda  = 0.;
	cd->lambda1 = 0.;
	cd->lambda2 = 0.;

	cd->w = NULL;

	cd->tolerance = 0.;

	cd->nrm1 = 0.;

	cd->b0 = 0.;
	cd->beta = NULL;
	cd->mu = NULL;
	cd->nu = NULL;

	cd->parallel = false;
	cd->total_iter = 0;

	cd->cfunc = NULL;

	cd->log10_lambda_upper = 0.;
	cd->log10_lambda_lower = 0.;
	cd->log10_dlambda = 0.;

	cd->output_fullpath = false;
	cd->output_info = false;

	cd->verbos = false;

	return cd;
}

/*** create new cdescent object ***/
cdescent *
cdescent_new (const double alpha, const linregmodel *lreg, const double tol, const int maxiter, bool parallel)
{
	cdescent	*cd;

	if (!lreg) error_and_exit ("cdescent_new", "linregmodel *lreg is empty.", __FILE__, __LINE__);
	if (alpha < 0. || 1. < alpha) error_and_exit ("cdescent_new", "alpha must be 0 <= alpha <= 1.", __FILE__, __LINE__);

	cd = cdescent_alloc ();
	if (cd == NULL) error_and_exit ("cdescent_new", "failed to allocate object.", __FILE__, __LINE__);

	cd->was_modified = false;

	cd->lreg = lreg;
	cd->m = &cd->lreg->y->m;
	cd->n = &cd->lreg->x->n;

	cd->tolerance = tol;

	cd->alpha1 = alpha;
	cd->alpha2 = 1. - alpha;

	/* if cd->lreg->d == NULL, regression type is Lasso */
	cd->is_regtype_lasso =  (cd->lreg->d == NULL);

	cd->lambda = (cd->alpha1 > 0.) ? cd->lreg->camax / cd->alpha1 : cd->lreg->camax;
	cd->lambda1 = cd->alpha1 * cd->lambda;
	cd->lambda2 = cd->alpha2 * cd->lambda;

	cd->beta = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, *cd->n, 1, *cd->n);
	mm_real_set_all (cd->beta, 0.);	// in initial, set to 0

	// mu = X * beta
	cd->mu = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, *cd->m, 1, *cd->m);
	mm_real_set_all (cd->mu, 0.);	// in initial, set to 0

	// nu = D * beta
	if (!cd->is_regtype_lasso) {
		cd->nu = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, cd->lreg->d->m, 1, cd->lreg->d->m);
		mm_real_set_all (cd->nu, 0.);	// in initial, set to 0
	}

	cd->maxiter = maxiter;
	cd->parallel = parallel;

	/* default values */
	log10_lambda_upper_default = floor (log10 (cd->lreg->camax)) + 1.;
	cd->log10_lambda_upper = log10_lambda_upper_default;
	if (cd->alpha1 > 0.) cd->log10_lambda_upper -= floor (log10 (cd->alpha1));
	cd->log10_lambda_lower = log10 (tol);
	cd->log10_dlambda = 0.1;
	strcpy (cd->fn_path, default_fn_path);	// default filename
	strcpy (cd->fn_info, default_fn_info);	// default filename

	return cd;
}

/*** free cdescent object ***/
void
cdescent_free (cdescent *cd)
{
	if (cd) {
		if (cd->w) mm_real_free (cd->w);
		if (cd->beta) mm_real_free (cd->beta);
		if (cd->mu) mm_real_free (cd->mu);
		if (cd->nu) mm_real_free (cd->nu);
		free (cd);
	}
	return;
}

void
cdescent_set_cyclic (cdescent *cd)
{
	cd->rule = CDESCENT_SELECTION_RULE_CYCLIC;
	return;
}

void
cdescent_set_stochastic (cdescent *cd, const unsigned int *seed)
{
	cd->rule = CDESCENT_SELECTION_RULE_STOCHASTIC;
	if (seed) srand (*seed);
	return;
}

/*** set penalty factor of adaptive L1 regression
 * penalty factor = w.^tau ***/
bool
cdescent_set_penalty_factor (cdescent *cd, const mm_dense *w, const double tau)
{
	int		j;
	if (w == NULL) return false;

	if (cd == NULL) error_and_exit ("cdescent_set_penalty_factor", "cdescent *cd is empty.", __FILE__, __LINE__);
	/* check whether w is dense general */
	if (!mm_real_is_dense (w)) error_and_exit ("cdescent_set_penalty_factor", "w must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (w)) error_and_exit ("cdescent_set_penalty_factor", "w must be general.", __FILE__, __LINE__);
	/* check whether w is vector */
	if (w->n != 1) error_and_exit ("cdescent_set_penalty_factor", "w must be vector.", __FILE__, __LINE__);
	/* check dimensions of x and w */
	if (w->m != *cd->n) error_and_exit ("cdescent_set_penalty_factor", "dimensions of w does not match.", __FILE__, __LINE__);

	/* copy w */
	if (!cd->w) cd->w = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, w->m, 1, w->nnz);
	if (fabs (tau - 1.) > DBL_EPSILON)	{
		for (j = 0; j < w->nnz; j++) cd->w->data[j] = pow (fabs (w->data[j]), tau);
	} else {
		for (j = 0; j < w->nnz; j++) cd->w->data[j] = fabs (w->data[j]);
	}
	return (cd->w != NULL);
}

void
cdescent_not_use_intercept (cdescent *cd)
{
	cd->use_intercept = false;
	return;
}

void
cdescent_set_constraint (cdescent *cd, constraint_func func)
{
	cd->cfunc = func;
	return;
}

static bool
is_regtype_l1 (const cdescent *cd)
{
	// whether alpha1 == 1
	return (fabs (cd->alpha1 - 1.) < DBL_EPSILON);
}

static bool
is_regtype_l2 (const cdescent *cd)
{
	// whether alpha2 == 1
	return (fabs (cd->alpha2 - 1.) < DBL_EPSILON);
}

void
cdescent_use_fixed_lambda (cdescent *cd, const double lambda)
{
	if (is_regtype_l1 (cd)) {	// L1 norm regularization
		cd->use_fixed_lambda = true;
		cd->lambda2 = lambda;
		cd->log10_lambda_upper = log10_lambda_upper_default;
	} else if (is_regtype_l2 (cd)) {	// L2 norm regularization
		cd->use_fixed_lambda = true;
		cd->lambda1 = lambda;
		cd->log10_lambda_upper = log10_lambda_upper_default;
	} else {
		printf_warning (
			"cdescent_use_fixed_lambda",
			"alpha must be 0 or 1, use_fixed_lambda is ignored",
			__FILE__, __LINE__);
	}
	return;
}

/*** set cd->lambda1
 * if designated lambda1 >= cd->lambda1_max, cd->lambda1 is set to cd->lambda1_max and return false
 * else cd->lambda1 is set to lambda1 and return true ***/
void
cdescent_set_lambda (cdescent *cd, const double lambda)
{
	cd->lambda = lambda;
	if (!cd->use_fixed_lambda) {
		cd->lambda1 = cd->alpha1 * lambda;
		cd->lambda2 = cd->alpha2 * lambda;
	} else {
		if (is_regtype_l1 (cd)) {	// L1 norm regularization
			cd->lambda1 = lambda;
		} else if (is_regtype_l2 (cd)) {	// L2 norm regularization
			cd->lambda2 = lambda;
		}
	}
	return;
}

/*** set cd->lambda1 to 10^log10_lambda1 ***/
void
cdescent_set_log10_lambda (cdescent *cd, const double log10_lambda)
{
	cdescent_set_lambda (cd, pow (10., log10_lambda));
	return;
}

/*** set cd->log10_lambda_upper ***/
void
cdescent_set_log10_lambda_upper (cdescent *cd, const double log10_lambda_upper)
{
	cd->log10_lambda_upper = log10_lambda_upper;
	return;
}

/*** set cd->log10_lambda_lower ***/
void
cdescent_set_log10_lambda_lower (cdescent *cd, const double log10_lambda_lower)
{
	cd->log10_lambda_lower = log10_lambda_lower;
	return;
}

/*** set cd->log10_dlambda ***/
void
cdescent_set_log10_dlambda (cdescent *cd, const double log10_dlambda)
{
	cd->log10_dlambda = log10_dlambda;
	return;
}

/*** set outputs_fullpath = true and specify output filename ***/
void
cdescent_set_outputs_fullpath (cdescent *cd, const char *fn)
{
	cd->output_fullpath = true;
	if (fn) strcpy (cd->fn_path, fn);
	return;
}

/*** set outputs_info = true and specify output filename ***/
void
cdescent_set_outputs_info (cdescent *cd, const char *fn)
{
	cd->output_info = true;
	if (fn) strcpy (cd->fn_info, fn);
	return;
}

double
cdescent_get_intercept_in_original_scale (const cdescent *cd)
{
	return cd->b0;
}

static void
beta_in_original_scale (mm_dense *beta, const double *xtx)
{
	int		j;
	for (j = 0; j < beta->m; j++) beta->data[j] /= sqrt (xtx[j]);
	return;
}

mm_dense *
cdescent_get_beta_in_original_scale (const cdescent *cd)
{
	mm_dense	*beta = mm_real_copy (cd->beta);
	if (cd->lreg->xnormalized && cd->lreg->xtx) {
		beta_in_original_scale (beta, cd->lreg->xtx);
	}
	return beta;
}

void
cdescent_init_beta (cdescent *cd, const mm_dense *beta)
{
	if (cd->beta) {
		int		n = *cd->n;
		mm_real_free (cd->beta);
		cd->beta = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n, 1, n);
	}
	mm_real_memcpy (cd->beta, beta);
	cd->nrm1 = mm_real_xj_asum (cd->beta, 0);
	mm_real_x_dot_y (false, 1., cd->lreg->x, cd->beta, 0., cd->mu);
	if (!cd->is_regtype_lasso) mm_real_x_dot_y (false, 1., cd->lreg->d, cd->beta, 0., cd->nu);
	return;
}
