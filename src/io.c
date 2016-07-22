/*
 * io.c
 *
 *  Created on: 2016/07/16
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "private/private.h"

/* output solution into FILE *stream
 * output format is
 * |beta| b0(intercept) beta[0] beta[1] ... */
static void
fprintf_solution (FILE *stream, const double b0, const mm_dense *beta)
{
	int		j;
	double	nrm1 = mm_real_xj_asum (beta, 0);
	fprintf (stream, "%.4e %.4e", nrm1, b0);
	for (j = 0; j < beta->m; j++) fprintf (stream, " %.8e", beta->data[j]);
	fprintf (stream, "\n");
	fflush (stream);
	return;
}

/* output solution path into FILE *stream
 * iter-th row of outputs indicates beta obtained by iter-th iteration */
void
fprintf_solutionpath (FILE *stream, cdescent *cd)
{
	double		b0 = cdescent_get_intercept_in_original_scale (cd);
	mm_dense	*beta = cdescent_get_beta_in_original_scale (cd);
	fprintf_solution (stream, b0, beta);
	mm_real_free (beta);
	return;
}
