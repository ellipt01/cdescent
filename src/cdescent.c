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

/* allocate pathwise optimization object */
static pathwise *
pathwise_alloc (void)
{
	pathwise	*path = (pathwise *) malloc (sizeof (pathwise));
	path->was_modified = false;

	path->log10_lambda1_upper = 0.;
	path->log10_lambda1_lower = 0.;
	path->dlog10_lambda1 = 0.;

	path->output_fullpath = false;
	path->output_bic_info = false;
	path->gamma_bic = 0.;
	path->index_opt = 0;
	path->b_opt = 0.;
	path->beta_opt = NULL;
	path->lambda1_opt = 0.;
	path->nrm1_opt = 0.;
	path->min_bic_val = CDESCENT_POSINF;
	return path;
}

/*** free pathwise optimization object ***/
static void
pathwise_free (pathwise *path)
{
	if (path) {
		if (path->beta_opt) mm_real_free (path->beta_opt);
		free (path);
	}
	return;
}

static reweighting_func *
reweighting_function_alloc (void)
{
	reweighting_func	*func = (reweighting_func *) malloc (sizeof (reweighting_func));
	func->tau = 0.;
	func->function = NULL;
	func->data = NULL;
	return func;
}

reweighting_func *
reweighting_function_new (const double tau, const weight_func function, void *data)
{
	reweighting_func	*func = reweighting_function_alloc ();
	func->tau = tau;
	func->function = function;
	func->data = data;
	return func;
}

static reweighting *
reweighting_alloc (void)
{
	reweighting	*rwt = (reweighting *) malloc (sizeof (reweighting));
	rwt->maxiter = 0;
	rwt->tolerance = 0.;
	rwt->func = NULL;
	return rwt;
}

/* allocate cdescent object */
static cdescent *
cdescent_alloc (void)
{
	cdescent	*cd = (cdescent *) malloc (sizeof (cdescent));
	if (cd == NULL) return NULL;

	cd->was_modified = false;

	cd->is_regtype_lasso = true;
	cd->update_intercept = true;
	cd->force_beta_nonnegative = false;

	cd->m = NULL;
	cd->n = NULL;
	cd->lreg = NULL;

	cd->lambda1 = 0.;
	cd->w = NULL;

	cd->tolerance = 0.;

	cd->nrm1 = 0.;

	cd->b = 0.;
	cd->beta = NULL;
	cd->mu = NULL;
	cd->nu = NULL;

	cd->parallel = false;
	cd->total_iter = 0;

	cd->path = NULL;
	cd->rwt = NULL;

	return cd;
}

/*** create new cdescent object ***/
cdescent *
cdescent_new (const linregmodel *lreg, const double tol, const int maxiter, bool parallel)
{
	cdescent	*cd;

	if (!lreg) error_and_exit ("cdescent_new", "linregmodel *lreg is empty.", __FILE__, __LINE__);

	cd = cdescent_alloc ();
	if (cd == NULL) error_and_exit ("cdescent_new", "failed to allocate object.", __FILE__, __LINE__);

	cd->was_modified = false;

	cd->lreg = lreg;
	cd->m = &cd->lreg->y->m;
	cd->n = &cd->lreg->x->n;

	/* if lreg->lambda2 == 0 || lreg->d == NULL, regression type is Lasso */
	cd->is_regtype_lasso =  (cd->lreg->lambda2 < DBL_EPSILON || cd->lreg->d == NULL);

	cd->tolerance = tol;

	cd->lambda1 = cd->lreg->camax;

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
	cd->path->log10_lambda1_upper = floor (log10 (cd->lreg->camax)) + 1.;
	cd->path->log10_lambda1_lower = log10 (tol);
	cd->path->dlog10_lambda1 = 0.1;
	strcpy (cd->path->fn_path, default_fn_path);	// default filename
	strcpy (cd->path->fn_bic, default_fn_bic);		// default filename

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
		if (cd->rwt) free (cd->rwt);
		free (cd);
	}
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

/*** set cd->lambda1
 * if designated lambda1 >= cd->lambda1_max, cd->lambda1 is set to cd->lambda1_max and return false
 * else cd->lambda1 is set to lambda1 and return true ***/
bool
cdescent_set_lambda1 (cdescent *cd, const double lambda1)
{
	if (cd->lreg->camax <= lambda1) {
		cd->lambda1 = cd->lreg->camax;
		return false;
	}
	cd->lambda1 = lambda1;
	return true;
}

/*** set cd->lambda1 to 10^log10_lambda1 ***/
bool
cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1)
{
	return cdescent_set_lambda1 (cd, pow (10., log10_lambda1));
}

void
cdescent_set_update_intercept (cdescent *cd, bool update_intercept)
{
	cd->update_intercept = update_intercept;
	return;
}

/*** routines to tuning pathwise CD optimization ***/
void
cdescent_set_pathwise_log10_lambda1_upper (cdescent *cd, const double log10_lambda1_upper)
{
	cd->path->log10_lambda1_upper = log10_lambda1_upper;
	return;
}

/*** routines to tuning pathwise CD optimization ***/
void
cdescent_set_pathwise_log10_lambda1_lower (cdescent *cd, const double log10_lambda1_lower)
{
	cd->path->log10_lambda1_lower = log10_lambda1_lower;
	return;
}

void
cdescent_set_pathwise_dlog10_lambda1 (cdescent *cd, const double dlog10_lambda1)
{
	cd->path->dlog10_lambda1 = dlog10_lambda1;
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

/*** set gamma of eBIC ***/
void
cdescent_set_pathwise_gamma_bic (cdescent *cd, const double gamma_bic)
{
	cd->path->gamma_bic = gamma_bic;
	return;
}

void
cdescent_set_reweighting (cdescent *cd, const int maxiter, const double tolerance, reweighting_func *func)
{
	cd->rwt = reweighting_alloc ();
	cd->rwt->maxiter = maxiter;
	cd->rwt->tolerance = tolerance;
	cd->rwt->func = func;
	return;
}
