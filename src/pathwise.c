/*
 * pathwise.c
 *
 *  Created on: 2014/10/31
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cdescent.h>
#include <pathwise.h>

#include "private.h"

// filename to output solution path
static const char	default_fn_path[] = "beta_path.data";

// filename to BIC info
static const char	default_fn_bic[] = "bic_info.data";

static pathwise *
pathwise_alloc (void)
{
	pathwise	*path = (pathwise *) malloc (sizeof (pathwise));
	path->output_fullpath = false;
	path->output_bic_info = false;
	path->gamma_bic = 0.;
	path->beta_opt = NULL;
	path->min_bic_val = CDESCENT_POSINF;
	return path;
}

pathwise *
pathwise_new (const double log10_lambda1_lower, const double dlog10_lambda1)
{
	pathwise	*path = pathwise_alloc ();
	strcpy (path->fn_path, default_fn_path);
	strcpy (path->fn_bic, default_fn_bic);
	path->log10_lambda1_lower = log10_lambda1_lower;
	path->dlog10_lambda1 = dlog10_lambda1;
	return path;
}

void
pathwise_free (pathwise *path)
{
	if (path) {
		if (path->beta_opt) mm_real_free (path->beta_opt);
		free (path);
	}
	return;
}

void
pathwise_output_fullpath (pathwise *path, const char *fn)
{
	path->output_fullpath = true;
	if (fn) strcpy (path->fn_path, fn);
	return;
}

void
pathwise_output_bic_info (pathwise *path, const char *fn)
{
	path->output_bic_info = true;
	if (fn) strcpy (path->fn_bic, fn);
	return;
}

void
pathwise_set_gamma_bic (pathwise *path, const double gamma_bic)
{
	path->gamma_bic = gamma_bic;
}
