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

/* cyclic.c */
extern bool			cdescent_do_update_once_cycle_cyclic (cdescent *cd);
/* stochastic.c */
extern bool			cdescent_do_update_once_cycle_stochastic (cdescent *cd);

/* utils.c */
extern void			fprintf_solutionpath (FILE *stream, cdescent *cd);

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

/* this function calculates and returns residual sum. of squares
 *   RSS = || y - X * beta ||^2 = || y - mu ||^2  */
static double
calc_rss (const cdescent *cd)
{
	double	rss;
	mm_dense	*r = mm_real_copy (cd->lreg->y);
	mm_real_axjpy (-1., cd->mu, 0, r);	// r = y - mu
	if (cd->use_intercept) mm_real_xj_add_const (r, 0, - cd->b0);	// r = y - mu - b0
	rss = mm_real_xj_ssq (r, 0);
	mm_real_free (r);
	return rss;
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

/*** do pathwise cyclic coordinate descent optimization.
 * The regression is starting at the smallest value lambda1_max for which
 * the entire vector beta = 0, and decreasing sequence of values for lambda1
 * while log10(lambda1) >= cd->log10_lambda1_lower.
 * log10(lambda1_max) is identical with log10 ( max ( abs(X' * y) ) ), where this
 * value is stored in cd->lreg->log10camax.
 * The interval of decreasing sequence on the log10 scale is cd->dlog10_lambda1. ***/
bool
cdescent_do_pathwise_optimization (cdescent *cd)
{
	int			iter;
	double		logt;
	bool		stop_flag = false;

	bool		converged;

	FILE		*fp_path = NULL;
	FILE		*fp_info = NULL;

	if (!cd) error_and_exit ("cdescent_do_pathwise_optimization", "cdescent *cd is empty.", __FILE__, __LINE__);

	// reset cdescent object if need
	if (cd->was_modified) cdescent_reset (cd);

	/* warm start */
	stop_flag = set_logt (cd->log10_lambda_lower, cd->log10_lambda_upper, &logt);

	if (cd->output_fullpath) {
		if ((fp_path = fopen (cd->fn_path, "w")) == NULL) {
			char	msg[80];
			sprintf (msg, "cannot open file %s.", cd->fn_path);
			printf_warning ("cdescent_do_pathwise_optimization", msg, __FILE__, __LINE__);
		}
		if (fp_path) fprintf_solutionpath (fp_path, cd);
	}

	if (cd->output_info) {
		if ((fp_info = fopen (cd->fn_info, "w")) == NULL) {
			char	msg[80];
			sprintf (msg, "cannot open file %s.", cd->fn_info);
			printf_warning ("cdescent_do_pathwise_optimization", msg, __FILE__, __LINE__);
		}
		// output regression info headers
		if (fp_info) fprintf (fp_info, "# nrm1\t\tnrm2\t\tRSS\t\tlambda1\t\tlambda2\n");
	}

	if (cd->verbos) fprintf (stderr, "starting pathwise optimization.\n");

	iter = 0;
	while (1) {

		cdescent_set_log10_lambda (cd, logt);
		if (cd->verbos) fprintf (stderr, "%d-th iteration lambda1 = %.4e, lamba2 = %.4e ", iter, cd->lambda1, cd->lambda2);

		if (!(converged = cdescent_do_update_one_cycle (cd))) break;

		// output solution path
		if (fp_path) fprintf_solutionpath (fp_path, cd);

		// output regression info
		if (fp_info) {
			// |beta|  ||beta||^2 RSS lambda1 lambda2
			fprintf (fp_info, "%.16e\t%.16e\t%.16e\t%.16e\t%.16e\n", cd->nrm1, mm_real_xj_ssq (cd->beta, 0), calc_rss (cd), cd->lambda1, cd->lambda2);
			fflush (fp_info);
		}

		if (cd->verbos) fprintf (stderr, "done.\n");

		if (stop_flag) break;

		/* if logt - dlog10_lambda1 < log10_lambda1, logt = log10_lambda1 and stop_flag is set to true
		 * else logt -= dlog10_lambda1 */
		stop_flag = set_logt (cd->log10_lambda_lower, logt - cd->log10_dlambda, &logt);

		iter++;
	}

	if (fp_path) fclose (fp_path);
	if (fp_info) fclose (fp_info);

	return converged;
}
