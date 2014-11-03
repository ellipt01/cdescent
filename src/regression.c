/*
 * update.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "private.h"
#include "atomic.h"

/* y += alpha * x(:,j) */
static void
axjpy (bool atomic, const double alpha, const mm_real *x, const int j, mm_dense *y)
{
	return (atomic) ? mm_real_axjpy_atomic (alpha, x, j, y) : mm_real_axjpy (alpha, x, j, y);
}

/* update intercept: (sum (y) - sum(X) * beta) / n */
static void
update_intercept (cdescent *cd)
{
	cd->b = 0.;
	// b += bar(y)
	if (!cd->lreg->ycentered) cd->b += cd->lreg->sy;
	// b -= bar(X) * beta
	if (!cd->lreg->xcentered) cd->b -= ddot_ (&cd->lreg->x->n, cd->lreg->sx, &ione, cd->beta->data, &ione);
	if (fabs (cd->b) > 0.) cd->b /= (double) cd->lreg->x->m;
	return;
}

/* update abs max: *amax = max (*amax, absval) */
static void
update_amax (bool atomic, double *amax, double absval)
{
	if (atomic) atomic_max (amax, absval);
	else if (*amax < absval) *amax = absval;
	return;
}

static void
cdescent_update (cdescent *cd, int j, bool atomic, double *amax_eta)
{
	// eta(j) = beta_new(j) - beta_prev(j)
	double	etaj = cdescent_beta_stepsize (cd, j);
	double	abs_etaj = fabs (etaj);

	if (abs_etaj > 0.) {
		// update beta: beta(j) += eta(j)
		cd->beta->data[j] += etaj;
		// update mu (= X * beta): mu += etaj * X(:,j)
		axjpy (atomic, etaj, cd->lreg->x, j, cd->mu);
		// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += etaj * D(:,j)
		if (!cd->lreg->is_regtype_lasso) axjpy (atomic, etaj, cd->lreg->d, j, cd->nu);
		// update max( |eta| )
		update_amax (atomic, amax_eta, abs_etaj);
	}
	return;
}

/*** progress coordinate descent update for one full cycle ***/
bool
cdescent_cyclic_update_once_cycle (cdescent *cd)
{
	int		j;
	double	amax_eta;	// max of |eta(j)| = |beta_new(j) - beta_prev(j)|

	/* b = (sum(y) - sum(X) * beta) / m */
	if (!cd->lreg->xcentered) update_intercept (cd);

	amax_eta = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent ***/
#ifdef _OPENMP
	if (cd->parallel) {
#pragma omp parallel for
		for (j = 0; j < cd->lreg->x->n; j++) cdescent_update (cd, j, true, &amax_eta);
	} else {
		for (j = 0; j < cd->lreg->x->n; j++) cdescent_update (cd, j, false, &amax_eta);
	}
#else
	for (j = 0; j < cd->lreg->x->n; j++) cdescent_update (cd, j, false, &amax_eta);
#endif

	cd->nrm1 = mm_real_xj_asum (cd->beta, 0);

	return (amax_eta < cd->tolerance);
}

/*** cyclic coordinate descent
 * repeat coordinate descent until solution is converged ***/
bool
cdescent_cyclic_update (cdescent *cd)
{
	int		iter = 0;
	bool	converged = false;

	while (!converged) {

		converged = cdescent_cyclic_update_once_cycle (cd);

		if (++iter >= cd->maxiter) {
			printf_warning ("cdescent_cyclic", "reaching max number of iterations.", __FILE__, __LINE__);
			break;
		}
	}
	cd->total_iter += iter;
	return converged;
}

// num of iterations. use for fprintf_solutionpath
static int			_iter_ = 0;

/* print solution path to stream */
static void
fprintf_solutionpath (FILE *stream, const cdescent *cd)
{
	int		j;
	fprintf (stream, "%d %.4e", _iter_++, cd->nrm1);
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
	if (new_logt <= logt_lower) {
		*logt = logt_lower;
		return true;
	}
	*logt = new_logt;
	return false;
}

/*** pathwise cyclic coordinate descent optimization.
 * The regression is starting at the smallest value λmax for which
 * the entire vector β = 0, and decreasing sequence of values for λ1 on
 * the log scale while log(λ1) >= path->log10_lambda1_lower.
 * log(λmax) is identical with log ( max ( abs(X' * y) ) ), where this
 * value is stored in cd->lreg->log10camax.
 * The interval of decreasing sequence on the log scale is path->dlog10_lambda1. ***/
void
cdescent_cyclic_pathwise (cdescent *cd, pathwiseopt *path)
{
	double	logt;
	bool	stop_flag = false;

	FILE	*fp_path = NULL;
	FILE	*fp_bic = NULL;

	/* warm start */
	stop_flag = set_logt (path->log10_lambda1_lower, cd->lreg->log10camax, &logt);

	if (path->output_fullpath) {
		if (!(fp_path = fopen (path->fn_path, "w"))) {
			char	msg[80];
			sprintf (msg, "cannot open file %s", path->fn_path);
			printf_warning ("cdescent_cyclic_pathwise", msg, __FILE__, __LINE__);
		}
	}
	if (path->output_bic_info) {
		if (!(fp_bic = fopen (path->fn_bic, "w"))) {
			char	msg[80];
			sprintf (msg, "cannot open file %s", path->fn_bic);
			printf_warning ("cdescent_cyclic_pathwise", msg, __FILE__, __LINE__);
		}
	}

	// output BIC info headers
	if (fp_bic) fprintf (fp_bic, "t\t\teBIC\t\tRSS\t\tdf\n");

	while (1) {
		bic_info	*info;

		cdescent_set_log10_lambda1 (cd, logt);

		if (!cdescent_cyclic_update (cd)) break;

		// output solution path
		if (fp_path) fprintf_solutionpath (fp_path, cd);

		// update optimal beta
		info = cdescent_eval_bic (cd, path->gamma_bic);
		if (info->bic_val < path->min_bic_val) {
			path->min_bic_val = info->bic_val;
			path->lambda1_opt = cd->lambda1;
			path->nrm1_opt = cd->nrm1;
			if (path->beta_opt) mm_real_free (path->beta_opt);
			path->beta_opt = mm_real_copy (cd->beta);

		}
		// output BIC info
		if (fp_bic) fprintf (fp_bic, "%.4e\t%.4e\t%.4e\t%.4e\n", cd->nrm1, info->bic_val, info->rss, info->df);
		free (info);

		if (stop_flag) break;

		/* if logt - dlog10_lambda1 < log10_lambda1, logt = log10_lambda1 and stop_flag is set to true
		 * else logt -= dlog10_lambda1 */
		stop_flag = set_logt (path->log10_lambda1_lower, logt - path->dlog10_lambda1, &logt);
	}

	if (fp_path) fclose (fp_path);
	if (fp_bic) fclose (fp_bic);

	return;
}
