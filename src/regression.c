/*
 * update.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "private/private.h"
#include "private/atomic.h"

/* stepsize.c */
extern double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* update intercept: (sum (y) - sum(X) * beta) / n */
static void
update_intercept (cdescent *cd)
{
	cd->b = 0.;
	// b += bar(y)
	if (!cd->lreg->ycentered) cd->b += *(cd->lreg->sy);
	// b -= bar(X) * beta
	if (!cd->lreg->xcentered) cd->b -= ddot_ (&cd->lreg->x->n, cd->lreg->sx, &ione, cd->beta->data, &ione);
	if (fabs (cd->b) > 0.) cd->b /= (double) cd->lreg->x->m;
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
	cd->beta->data[j] += etaj;
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
	cd->beta->data[j] += etaj;
	// update mu (= X * beta): mu += etaj * X(:,j)
	mm_real_axjpy_atomic (etaj, cd->lreg->x, j, cd->mu);
	// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += etaj * D(:,j)
	if (!cd->is_regtype_lasso) mm_real_axjpy_atomic (etaj, cd->lreg->d, j, cd->nu);
	// update max( |etaj| )
	atomic_max (amax_eta, abs_etaj);

	return;
}

/*** progress coordinate descent update for one full cycle ***/
bool
cdescent_cyclic_update_once_cycle (cdescent *cd)
{
	int		j;
	double	amax_eta;	// max of |eta(j)| = |beta_new(j) - beta_prev(j)|

	if (!cd) error_and_exit ("cdescent_cyclic_update_once_cycle", "cdescent *cd is empty.", __FILE__, __LINE__);

	/* b = (sum(y) - sum(X) * beta) / m */
	if (!cd->lreg->xcentered) update_intercept (cd);

	amax_eta = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent
	 * the following code was referring to shotgun by A.Kyrola,
	 * https://github.com/akyrola/shotgun ****/
	if (cd->parallel) {
#pragma omp parallel for
		for (j = 0; j < cd->lreg->x->n; j++) cdescent_update_atomic (cd, j, &amax_eta);
	} else {
		for (j = 0; j < cd->lreg->x->n; j++) cdescent_update (cd, j, &amax_eta);
	}

	cd->nrm1 = mm_real_xj_asum (cd->beta, 0);

	if (!cd->was_modified) cd->was_modified = true;

	return (amax_eta < cd->tolerance);
}

/*** cyclic coordinate descent
 * repeat coordinate descent until solution is converged ***/
bool
cdescent_cyclic_update (cdescent *cd)
{
	int		iter = 0;
	bool	converged = false;

	if (!cd) error_and_exit ("cdescent_cyclic_update", "cdescent *cd is empty.", __FILE__, __LINE__);

	while (!converged) {

		converged = cdescent_cyclic_update_once_cycle (cd);

		if (++iter >= cd->maxiter) {
			printf_warning ("cdescent_cyclic_update", "reaching max number of iterations.", __FILE__, __LINE__);
			break;
		}
	}
	cd->total_iter += iter;
	return converged;
}

/* print out solution path to FILE *stream */
static void
fprintf_solutionpath (FILE *stream, const int iter, const cdescent *cd)
{
	int		j;
	fprintf (stream, "%d %.4e", iter, cd->nrm1);
	for (j = 0; j < cd->beta->m; j++) fprintf (stream, " %.4e", cd->beta->data[j]);
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
store_optimal (pathwiseopt *path, const double lambda1, const double nrm1, const mm_dense *beta)
{
	path->lambda1_opt = lambda1;
	path->nrm1_opt = nrm1;
	if (path->beta_opt) mm_real_free (path->beta_opt);
	path->beta_opt = mm_real_copy (beta);
	return;
}

/*** pathwise cyclic coordinate descent optimization.
 * The regression is starting at the smallest value λ1max for which
 * the entire vector β = 0, and decreasing sequence of values for λ1
 * while log10(λ1) >= path->log10_lambda1_lower.
 * log10(λ1max) is identical with log10 ( max ( abs(X' * y) ) ), where this
 * value is stored in cd->lreg->log10camax.
 * The interval of decreasing sequence on the log10 scale is path->dlog10_lambda1. ***/
bool
cdescent_cyclic_pathwise (cdescent *cd, pathwiseopt *path)
{
	int		iter = 0;
	double	logt;
	bool	stop_flag = false;

	FILE	*fp_path = NULL;
	FILE	*fp_bic = NULL;

	if (!cd) error_and_exit ("cdescent_cyclic_pathwise", "cdescent *cd is empty.", __FILE__, __LINE__);
	if (!path) error_and_exit ("cdescent_cyclic_pathwise", "pathwiseopt *path is empty.", __FILE__, __LINE__);

	// initialize cdescent if need
	if (cd->was_modified) {
		cd->nrm1 = 0.;
		cd->b = 0.;
		mm_real_set_all (cd->beta, 0.);
		mm_real_set_all (cd->mu, 0.);
		if (!cd->is_regtype_lasso) mm_real_set_all (cd->nu, 0.);
		cd->total_iter = 0;
		cd->was_modified = false;
	}
	// initialize pathwiseopt if need
	if (path->was_modified) {
		if (path->beta_opt) mm_real_free (path->beta_opt);
		path->beta_opt = NULL;
		path->lambda1_opt = 0.;
		path->nrm1_opt = 0.;
		path->min_bic_val = CDESCENT_POSINF;
		path->was_modified = false;
	}

	/* warm start */
	stop_flag = set_logt (path->log10_lambda1_lower, cd->lreg->log10camax, &logt);

	if (path->output_fullpath) {
		if (!(fp_path = fopen (path->fn_path, "w"))) {
			char	msg[80];
			sprintf (msg, "cannot open file %s.", path->fn_path);
			printf_warning ("cdescent_cyclic_pathwise", msg, __FILE__, __LINE__);
		}
		if (fp_path) fprintf_solutionpath (fp_path, iter, cd);
	}
	if (path->output_bic_info) {
		if (!(fp_bic = fopen (path->fn_bic, "w"))) {
			char	msg[80];
			sprintf (msg, "cannot open file %s.", path->fn_bic);
			printf_warning ("cdescent_cyclic_pathwise", msg, __FILE__, __LINE__);
		}
		// output BIC info headers
		if (fp_bic) fprintf (fp_bic, "t\t\teBIC\t\tRSS\t\tdf\n");
	}

	while (1) {
		bic_info	*info;

		if (path->func) {
			mm_dense	*w = path->func->function (iter, cd, path->func->data);
			cdescent_set_penalty_factor (cd, w, path->func->tau);
			mm_real_free (w);
		}


		cdescent_set_log10_lambda1 (cd, logt);

		if (!cdescent_cyclic_update (cd)) break;

		// output solution path
		if (fp_path) fprintf_solutionpath (fp_path, iter, cd);

		info = cdescent_eval_bic (cd, path->gamma_bic);
		// if bic_val < min_bic_val, update min_bic_val, lambda1_opt, nrm1_opt and beta_opt
		if (info->bic_val < path->min_bic_val) {
			store_optimal (path, cd->lambda1, cd->nrm1, cd->beta);
			path->min_bic_val = info->bic_val;
			if (!path->was_modified) path->was_modified = true;
		}

		// output BIC info
		if (fp_bic) fprintf (fp_bic, "%.4e\t%.4e\t%.4e\t%.4e\n", cd->nrm1, info->bic_val, info->rss, info->df);
		free (info);

		if (stop_flag) break;

		/* if logt - dlog10_lambda1 < log10_lambda1, logt = log10_lambda1 and stop_flag is set to true
		 * else logt -= dlog10_lambda1 */
		stop_flag = set_logt (path->log10_lambda1_lower, logt - path->dlog10_lambda1, &logt);

		iter++;
	}

	if (fp_path) fclose (fp_path);
	if (fp_bic) fclose (fp_bic);

	return path->was_modified;
}
