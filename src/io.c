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

/* update.c */
extern mm_dense	*unscaled_beta (cdescent *cd);

static void
fprintf_solution (FILE *stream, const double b0, const mm_dense *beta)
{
	int		j;
	fprintf (stream, "%.4e %.4e", mm_real_xj_asum (beta, 0), b0);
	for (j = 0; j < beta->m; j++) fprintf (stream, " %.8e", beta->data[j]);
	fprintf (stream, "\n");
	fflush (stream);
	return;
}

/* fprint solution path to FILE *stream
 * iter-th row of outputs indicates beta obtained by iter-th iteration */
void
fprintf_solutionpath (FILE *stream, cdescent *cd)
{
	if (cd->lreg->xnormalized) {
		mm_dense	*beta = unscaled_beta (cd);
		fprintf_solution (stream, cd->b0, beta);
		mm_real_free (beta);
	} else {
		fprintf_solution (stream, cd->b0, cd->beta);
	}
	return;
}
