/*
 * update.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>
#include <mmreal.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "private/private.h"
#include "private/atomic.h"

/* stepsize.c */
extern double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* update intercept: (sum (y) - sum(X) * beta) / n */
static void
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
update_betaj (bool nonnegative, double *betaj, double *etaj, double *abs_etaj)
{
	double	betaj_val = *betaj;
	// nonnegative coordinate descent after Franc, Hlavac and Navara, 2005.
	if (nonnegative && betaj_val + *etaj < 0.) {
		*etaj = - betaj_val;
		*abs_etaj = fabs (*etaj);
		*betaj = 0.;
	} else *betaj += *etaj;
	return;
}

/* update beta, mu, nu and amax_eta */
static void
cdescent_update (cdescent *cd, int j, double *amax_eta)
{
	// eta(j) = beta_new(j) - beta_prev(j)
	double	etaj = cdescent_beta_stepsize (cd, j);
	double	abs_etaj = fabs (etaj);

	if (abs_etaj < DBL_EPSILON) return;

	// update beta: beta(j) += eta(j)
	update_betaj (cd->force_beta_nonnegative, &cd->beta->data[j], &etaj, &abs_etaj);
	// update mu (= X * beta): mu += eta(j) * X(:,j)
	mm_real_axjpy (etaj, cd->lreg->x, j, cd->mu);
	// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += eta(j) * D(:,j)
	if (!cd->is_regtype_lasso) mm_real_axjpy (etaj, cd->lreg->d, j, cd->nu);
	// update max( |eta| )
	if (*amax_eta < abs_etaj) *amax_eta = abs_etaj;

	return;
}

/* update beta, mu, nu and amax_eta in atomic */
static void
cdescent_update_atomic (cdescent *cd, int j, double *amax_eta)
{
	// eta(j) = beta_new(j) - beta_prev(j)
	double	etaj = cdescent_beta_stepsize (cd, j);
	double	abs_etaj = fabs (etaj);

	if (abs_etaj < DBL_EPSILON) return;

	// update beta: beta(j) += etaj
	update_betaj (cd->force_beta_nonnegative, &cd->beta->data[j], &etaj, &abs_etaj);
	// update mu (= X * beta): mu += etaj * X(:,j)
	mm_real_axjpy_atomic (etaj, cd->lreg->x, j, cd->mu);
	// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += etaj * D(:,j)
	if (!cd->is_regtype_lasso) mm_real_axjpy_atomic (etaj, cd->lreg->d, j, cd->nu);
	// update max( |etaj| )
	atomic_max (amax_eta, abs_etaj);

	return;
}

/*** progress coordinate descent update for one full cycle ***/
static bool
cdescent_update_once_cycle (cdescent *cd)
{
	int		j;
	int		n = *cd->n;
	double	amax_eta;	// max of |eta(j)| = |beta_new(j) - beta_prev(j)|

	/* b = (sum(y) - sum(X) * beta) / m */
	if (cd->use_intercept && !cd->lreg->xcentered) update_intercept (cd);

	amax_eta = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent
	 * the following code was referring to shotgun by A.Kyrola,
	 * https://github.com/akyrola/shotgun ****/
	if (cd->parallel) {
#pragma omp parallel for
		for (j = 0; j < n; j++) cdescent_update_atomic (cd, j, &amax_eta);
	} else {
		for (j = 0; j < n; j++) cdescent_update (cd, j, &amax_eta);
	}

	cd->nrm1 = mm_real_xj_asum (cd->beta, 0);

	if (!cd->was_modified) cd->was_modified = true;

	return (amax_eta < cd->tolerance);
}

/*** do cyclic coordinate descent optimization for fixed lambda1
 * repeat coordinate descent algorithm until solution is converged ***/
bool
cdescent_do_cyclic_update (cdescent *cd)
{
	int		ccd_iter = 0;
	bool	converged = false;

	if (!cd) error_and_exit ("cdescent_do_cyclic_update", "cdescent *cd is empty.", __FILE__, __LINE__);

	while (!converged) {

		converged = cdescent_update_once_cycle (cd);

		if (++ccd_iter >= cd->maxiter) {
			printf_warning ("cdescent_do_cyclic_update", "reaching max number of iterations.", __FILE__, __LINE__);
			break;
		}

	}
	cd->total_iter += ccd_iter;
	return converged;
}

/*** fprint solution path to FILE *stream
 * iter-th row of outputs indicates beta obtained by iter-th iteration ***/
static void
fprintf_solutionpath (FILE *stream, const int iter, const mm_dense *beta)
{
	int		j;
	fprintf (stream, "%d %.4e", iter, mm_real_xj_asum (beta, 0));
	for (j = 0; j < beta->m; j++) fprintf (stream, " %.4e", beta->data[j]);
	fprintf (stream, "\n");
	return;
}

/* set log(t):
 * if new_logt <= logt_lower, *logt = logt_lower and return true
 * else *logt = new_logt */
static bool
set_logt (const double logt_lower, const double new_logt, double *logt)
{
	bool	reachs_lower = false;
	if (new_logt <= logt_lower) {
		*logt = logt_lower;
		reachs_lower = true;
	} else *logt = new_logt;

	return reachs_lower;
}

/* store lambda1_opt, nrm1_opt and beta_opt */
static void
store_optimal (cdescent *cd, const int index)
{
	cd->path->index_opt = index;
	cd->path->lambda1_opt = cd->lambda1;
	cd->path->nrm1_opt = cd->nrm1;
	if (cd->use_intercept) cd->path->b0_opt = cd->b0;
	if (cd->path->beta_opt) mm_real_free (cd->path->beta_opt);
	cd->path->beta_opt = mm_real_copy (cd->beta);
	return;
}

/* reset cdescent object */
static void
cdescent_reset (cdescent *cd)
{
	cd->nrm1 = 0.;
	if (cd->use_intercept) cd->b0 = 0.;
	mm_real_set_all (cd->beta, 0.);
	mm_real_set_all (cd->mu, 0.);
	if (!cd->is_regtype_lasso) mm_real_set_all (cd->nu, 0.);
	cd->total_iter = 0;
	cd->was_modified = false;
	return;
}

/* reset pathwise object */
static void
pathwise_reset (pathwise *path)
{
	if (path->beta_opt) mm_real_free (path->beta_opt);
	path->beta_opt = NULL;
	path->lambda1_opt = 0.;
	path->nrm1_opt = 0.;
	path->min_bic_val = CDESCENT_POSINF;
	path->was_modified = false;
	return;
}

/*** do reweighted coordinate descent optimization.
 * The weight for L1 norm is updated by calling reweighting function
 * cd->path->func using beta of the previous iteration. ***/
bool
cdescent_do_reweighting (cdescent *cd)
{
	int			iter;
	double		dnrm1;
	bool		converged = true;

	if (!cd->rwt) error_and_exit ("cdescent_do_reweighting", "cd->rwt is empty.", __FILE__, __LINE__);
	if (!cd->rwt->func) error_and_exit ("cdescent_do_reweighting", "cd->rwt->func is empty.", __FILE__, __LINE__);

	iter = 0;
	do {
		double		nrm1_prev = cd->nrm1;
		mm_dense	*w = cd->rwt->func->function (cd, cd->rwt->func->data);
		cdescent_set_penalty_factor (cd, w, cd->rwt->func->tau);
		mm_real_free (w);

		if (!(converged = cdescent_do_cyclic_update (cd))) break;

		if (++iter > cd->rwt->maxiter) {
			printf_warning ("cdescent_do_reweighting", "reaching max number of iterations.", __FILE__, __LINE__);
			break;
		}

		dnrm1 = fabs (cd->nrm1 - nrm1_prev);
	} while (dnrm1 > cd->rwt->tolerance);

	return converged;
}

/*** do pathwise cyclic coordinate descent optimization.
 * The regression is starting at the smallest value lambda1_max for which
 * the entire vector beta = 0, and decreasing sequence of values for lambda1
 * while log10(lambda1) >= path->log10_lambda1_lower.
 * log10(lambda1_max) is identical with log10 ( max ( abs(X' * y) ) ), where this
 * value is stored in cd->lreg->log10camax.
 * The interval of decreasing sequence on the log10 scale is path->dlog10_lambda1. ***/
bool
cdescent_do_pathwise_optimization (cdescent *cd)
{
	int			iter = 0;
	double		logt;
	bool		stop_flag = false;

	bool		converged;

	FILE		*fp_path = NULL;
	FILE		*fp_bic = NULL;

	if (!cd) error_and_exit ("cdescent_do_pathwise_optimization", "cdescent *cd is empty.", __FILE__, __LINE__);
	if (!cd->path) error_and_exit ("cdescent_do_pathwise_optimization", "cd->path is empty.", __FILE__, __LINE__);

	// reset cdescent object if need
	if (cd->was_modified) cdescent_reset (cd);
	// reset pathwise object if need
	if (cd->path->was_modified) pathwise_reset (cd->path);

	/* warm start */
	stop_flag = set_logt (cd->path->log10_lambda_lower, cd->path->log10_lambda_upper, &logt);

	if (cd->path->output_fullpath) {
		if (!(fp_path = fopen (cd->path->fn_path, "w"))) {
			char	msg[80];
			sprintf (msg, "cannot open file %s.", cd->path->fn_path);
			printf_warning ("cdescent_do_pathwise_optimization", msg, __FILE__, __LINE__);
		}
		if (fp_path) fprintf_solutionpath (fp_path, iter, cd->beta);
	}
	if (cd->path->output_bic_info) {
		if (!(fp_bic = fopen (cd->path->fn_bic, "w"))) {
			char	msg[80];
			sprintf (msg, "cannot open file %s.", cd->path->fn_bic);
			printf_warning ("cdescent_do_pathwise_optimization", msg, __FILE__, __LINE__);
		}
		// output BIC info headers
		if (fp_bic) fprintf (fp_bic, "t\t\teBIC\t\tRSS\t\tdf\n");
	}

	while (1) {
		bic_info	*info;

		iter++;

		cdescent_set_log10_lambda (cd, logt);

		if (!(converged = cdescent_do_cyclic_update (cd))) break;

		// reweighting
		if (cd->rwt && !(converged = cdescent_do_reweighting (cd))) break;

		// output solution path
		if (fp_path) fprintf_solutionpath (fp_path, iter, cd->beta);

		info = cdescent_eval_bic (cd, cd->path->gamma_bic);
		// if bic_val < min_bic_val, update min_bic_val, lambda1_opt, nrm1_opt and beta_opt
		if (info->bic_val < cd->path->min_bic_val) {
			store_optimal (cd, iter);
			cd->path->min_bic_val = info->bic_val;
			if (!cd->path->was_modified) cd->path->was_modified = true;
		}

		// output BIC info
		if (fp_bic) fprintf (fp_bic, "%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n", cd->nrm1, info->bic_val, info->rss, info->df, mm_real_xj_ssq (cd->beta, 0), cd->lambda2);
		free (info);

		if (stop_flag) break;

		/* if logt - dlog10_lambda1 < log10_lambda1, logt = log10_lambda1 and stop_flag is set to true
		 * else logt -= dlog10_lambda1 */
		stop_flag = set_logt (cd->path->log10_lambda_lower, logt - cd->path->dlog10_lambda, &logt);

	}

	if (fp_path) fclose (fp_path);
	if (fp_bic) fclose (fp_bic);

	return converged;
}
