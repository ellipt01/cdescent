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

#include "private/private.h"

/* cyclic.c */
extern bool			cdescent_do_update_once_cycle_cyclic (cdescent *cd);
/* stochastic.c */
extern bool			cdescent_do_update_once_cycle_stochastic (cdescent *cd);

typedef bool (*update_one_cycle) (cdescent *cd);

/*** do cyclic coordinate descent optimization for fixed lambda1
 * repeat coordinate descent algorithm until solution is converged ***/
bool
cdescent_do_update_one_cycle (cdescent *cd)
{
	int					ccd_iter = 0;
	bool				converged = false;

	update_one_cycle	update_func;

	update_func = (cd->rule == CDESCENT_SELECTION_RULE_STOCHASTIC) ?
			cdescent_do_update_once_cycle_stochastic : cdescent_do_update_once_cycle_cyclic;


	if (!cd) error_and_exit ("cdescent_do_cyclic_update", "cdescent *cd is empty.", __FILE__, __LINE__);

	while (!converged) {

		converged = update_func (cd);

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
	fflush (stream);
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
	cd->path->lambda_opt = cd->lambda;
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
	path->b0_opt = 0.;
	path->lambda_opt = 0.;
	path->nrm1_opt = 0.;
	path->min_bic_val = CDESCENT_POSINF;
	path->was_modified = false;
	return;
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
		if (fp_bic) fprintf (fp_bic, "# nrm1\t\tBIC\t\tRSS\t\tdf\t\tnrm2\t\tlambda1\tlambda2\n");
	}

	if (cd->path->verbos) fprintf (stderr, "starting pathwise optimization.\n");

	while (1) {
		bic_info	*info;

		iter++;

		cdescent_set_log10_lambda (cd, logt);
		if (cd->path->verbos) fprintf (stderr, "%d-th iteration lambda1 = %.4e, lamba2 = %.4e ", iter, cd->lambda1, cd->lambda2);


		if (!(converged = cdescent_do_update_one_cycle (cd))) break;

		// output solution path
		if (fp_path) fprintf_solutionpath (fp_path, iter, cd->beta);

		info = cdescent_eval_bic (cd);
		// if bic_val < min_bic_val, update min_bic_val, lambda1_opt, nrm1_opt and beta_opt
		if (info->bic_val < cd->path->min_bic_val) {
			store_optimal (cd, iter);
			cd->path->min_bic_val = info->bic_val;
			if (!cd->path->was_modified) cd->path->was_modified = true;
		}
		if (cd->path->verbos) fprintf (stderr, "bic = %.4e ... ", info->bic_val);
		// output BIC info
		if (fp_bic) {
			// |beta|  BIC  RSS  df  ||beta||^2 lambda2
			fprintf (fp_bic, "%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n", cd->nrm1, info->bic_val, info->rss, info->df, mm_real_xj_ssq (cd->beta, 0), cd->lambda1, cd->lambda2);
			fflush (fp_bic);
		}
		free (info);

		if (stop_flag) break;

		/* if logt - dlog10_lambda1 < log10_lambda1, logt = log10_lambda1 and stop_flag is set to true
		 * else logt -= dlog10_lambda1 */
		stop_flag = set_logt (cd->path->log10_lambda_lower, logt - cd->path->dlog10_lambda, &logt);

		if (cd->path->verbos) fprintf (stderr, "done.\n");

	}

	if (fp_path) fclose (fp_path);
	if (fp_bic) fclose (fp_bic);

	return converged;
}
