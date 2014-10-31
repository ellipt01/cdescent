/*
 * pathwise.c
 *
 *  Created on: 2014/10/31
 *      Author: utsugi
 */

#include <stdio.h>
#include <cdescent.h>

#include "private.h"

// num of iterations
int			iter = 0;

// default value of gamma_bic
double		gamma_bic = 0.;

// solution path
bool 		output_solutionpath = false;
const char	fn_path[] = "beta_path.data";

// bic info
bool		output_bic_info = false;
const char	fn_bic[] = "bic_info.data";

/*** set conditions of pathwise cyclic coordinate descent algorithm.
 * bool output_solutionpath:	if this is true, entire solution path is output to the file "beta_path.data"
 * bool output_bic_info:			if this is true, bic info is output to the file "bic_info.data"
 * const double gamma_bic_val:	gamma of eBIC (default is 0.) ***/
void
cdescent_cyclic_pathwise_set_conditions (bool is_output_solutionpath, bool is_output_bic_info, const double gamma_bic_val)
{
	// whether output solution path
	output_solutionpath = is_output_solutionpath;
	output_bic_info = is_output_bic_info;
	// set gamma for eBIC
	if (gamma_bic_val < 0.) {
		printf_warning ("cdescent_cyclic_pathwise_set_gamma_bic", "gamma of eBIC must be >= 0. gamma is set to 0.\n", __FILE__, __LINE__);
		gamma_bic = 0.;
	} else gamma_bic = gamma_bic_val;

	return;
}

/* print solution path to stream */
static void
fprintf_solutionpath (FILE *stream, const cdescent *cd)
{
	int		j;
	fprintf (stream, "%d %.4e", iter++, cd->nrm1);
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

/* update cd->lambda1_opt, cd->nrm1_opt and cd->beta_opt */
static void
update_beta_optimal (cdescent *cd, const double lambda1, const double nrm1, const mm_dense *beta)
{
	cd->lambda1_opt = lambda1;
	cd->nrm1_opt = nrm1;
	if (cd->beta_opt) mm_real_free (cd->beta_opt);
	cd->beta_opt = mm_real_copy (beta);
	return;
}

/*** pathwise cyclic coordinate descent algorithm.
 * The regression is starting at the smallest value λmax for which
 * the entire vector β = 0, and decreasing sequence of values for λ1 on
 * the log scale while log(λ1) >= log10_lambda1_lower.
 * log(λmax) is identical with log ( max ( abs(X' * y) ) ), where this
 * value is stored in cd->lreg->log10camax.
 * The interval of decreasing sequence on the log scale is dlog10_lambda1. ***/
void
cdescent_cyclic_pathwise (cdescent *cd, const double log10_lambda1_lower, const double dlog10_lambda1)
{
	double	logt;
	bool	stop_flag = false;

	double	bic_min = + 1. / 0.;
	FILE	*fp_path = NULL;
	FILE	*fp_bic = NULL;

	/* warm start */
	stop_flag = set_logt (log10_lambda1_lower, cd->lreg->log10camax, &logt);

	if (output_solutionpath) fp_path = fopen (fn_path, "w");
	if (output_bic_info) fp_bic = fopen (fn_bic, "w");

	// output bic info
	if (fp_bic) fprintf (fp_bic, "t\tebic\trss\tdf\n");

	while (1) {
		bic_info	*info;

		cdescent_set_log10_lambda1 (cd, logt);

		if (!cdescent_cyclic_update (cd)) break;

		// output solution path
		if (fp_path) fprintf_solutionpath (fp_path, cd);

		// update optimal beta
		info = cdescent_eval_bic (cd, gamma_bic);
		if (info->bic_val < bic_min) {
			update_beta_optimal (cd, cd->lambda1, cd->nrm1, cd->beta);
			bic_min = info->bic_val;
		}
		// output bic info
		if (fp_bic) fprintf (fp_bic, "%.4e\t%.4e\t%.4e\t%.4e\n", cd->nrm1, info->bic_val, info->rss, info->df);
		bic_info_free (info);

		if (stop_flag) break;

		/* if logt - dlog10_lambda1 < log10_lambda1, logt = log10_lambda1 and stop_flag is set to true
		 * else logt -= dlog10_lambda1 */
		stop_flag = set_logt (log10_lambda1_lower, logt - dlog10_lambda1, &logt);
	}

	if (fp_path) fclose (fp_path);
	if (fp_bic) fclose (fp_bic);

	return;
}
