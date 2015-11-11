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

// default file to output BIC info
static const char	default_fn_bic[] = "bic_info.data";

/* bic.c */
extern double	cdescent_default_bic_eval_func (const cdescent *cd, bic_info *info, void *data);
extern bic_info	*calc_bic_info (const cdescent *cd);

double			log10_lambda_upper_default = 0.;

/**********************************
 *  pathwise optimization object  *
 **********************************/

/* allocate pathwise optimization object */
static pathwise *
pathwise_alloc (void)
{
	pathwise	*path = (pathwise *) malloc (sizeof (pathwise));
	path->was_modified = false;

	path->log10_lambda_upper = 0.;
	path->log10_lambda_lower = 0.;
	path->dlog10_lambda = 0.;

	path->output_fullpath = false;
	path->output_bic_info = false;

	path->bicfunc = NULL;

	path->index_opt = 0;
	path->b0_opt = 0.;
	path->beta_opt = NULL;
	path->lambda_opt = 0.;
	path->nrm1_opt = 0.;
	path->min_bic_val = CDESCENT_POSINF;

	path->verbos = false;

	return path;
}

/* free pathwise optimization object */
static void
pathwise_free (pathwise *path)
{
	if (path) {
		if (path->beta_opt) mm_real_free (path->beta_opt);
		free (path);
	}
	return;
}

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
	cd->use_fixed_lambda2 = false;

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

	cd->path = NULL;
	cd->cfunc = NULL;

	cd->rule = CDESCENT_SELECTION_RULE_CYCLIC;

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

	cd->path = pathwise_alloc ();
	/* default values */
	log10_lambda_upper_default = floor (log10 (cd->lreg->camax)) + 1.;
	cd->path->log10_lambda_upper = log10_lambda_upper_default;
	if (cd->alpha1 > 0.) cd->path->log10_lambda_upper -= floor (log10 (cd->alpha1));
	cd->path->log10_lambda_lower = log10 (tol);
	cd->path->dlog10_lambda = 0.1;
	strcpy (cd->path->fn_path, default_fn_path);	// default filename
	strcpy (cd->path->fn_bic, default_fn_bic);		// default filename
	// default bic evaluation function
	cd->path->bicfunc = bic_function_new (cdescent_default_bic_eval_func, NULL);

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
		if (cd->path) pathwise_free (cd->path);
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

void
cdescent_use_fixed_lambda2 (cdescent *cd, const double lambda2)
{
	cd->use_fixed_lambda2 = true;
	cd->lambda2 = lambda2;

	cd->alpha1 = 1.;
	cd->alpha2 = 0.;
	cd->path->log10_lambda_upper = log10_lambda_upper_default;
	return;
}

/*** set cd->lambda1
 * if designated lambda1 >= cd->lambda1_max, cd->lambda1 is set to cd->lambda1_max and return false
 * else cd->lambda1 is set to lambda1 and return true ***/
void
cdescent_set_lambda (cdescent *cd, const double lambda)
{
	cd->lambda = lambda;
	cd->lambda1 = cd->alpha1 * lambda;
	if (!cd->use_fixed_lambda2) cd->lambda2 = cd->alpha2 * lambda;
	return;
}

/*** set cd->lambda1 to 10^log10_lambda1 ***/
void
cdescent_set_log10_lambda (cdescent *cd, const double log10_lambda)
{
	cdescent_set_lambda (cd, pow (10., log10_lambda));
	return;
}

/*** routines to tuning pathwise CD optimization ***/
void
cdescent_set_pathwise_log10_lambda_upper (cdescent *cd, const double log10_lambda_upper)
{
	cd->path->log10_lambda_upper = log10_lambda_upper;
	return;
}

/*** routines to tuning pathwise CD optimization ***/
void
cdescent_set_pathwise_log10_lambda_lower (cdescent *cd, const double log10_lambda_lower)
{
	cd->path->log10_lambda_lower = log10_lambda_lower;
	return;
}

void
cdescent_set_pathwise_dlog10_lambda (cdescent *cd, const double dlog10_lambda)
{
	cd->path->dlog10_lambda = dlog10_lambda;
	return;
}

/*** set outputs_fullpath = true and specify output filename ***/
void
cdescent_set_pathwise_outputs_fullpath (cdescent *cd, const char *fn)
{
	cd->path->output_fullpath = true;
	if (fn) strcpy (cd->path->fn_path, fn);
	return;
}

/*** set outputs_bic_info = true and specify output filename ***/
void
cdescent_set_pathwise_outputs_bic_info (cdescent *cd, const char *fn)
{
	cd->path->output_bic_info = true;
	if (fn) strcpy (cd->path->fn_bic, fn);
	return;
}

/*** evaluation of BIC ***/

/*** set bic function of pathwise object ***/
void
cdescent_set_pathwise_bic_func (cdescent *cd, bic_func *func)
{
	if (cd->path->bicfunc) free (cd->path->bicfunc);
	cd->path->bicfunc = func;
	return;
}

/*** evaluate BIC ***/
bic_info *
cdescent_eval_bic (const cdescent *cd)
{
	return calc_bic_info (cd);
}
