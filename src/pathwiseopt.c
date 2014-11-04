/*
 * pathwiseopt.c
 *
 *  Created on: 2014/10/31
 *      Author: utsugi
 */

#include <stdlib.h>
#include <string.h>
#include <cdescent.h>

#include "private/private.h"

// default file to output solution path
static const char	default_fn_path[] = "beta_path.data";

// default file to output BIC info
static const char	default_fn_bic[] = "bic_info.data";

/* allocate pathwise optimization object */
static pathwiseopt *
pathwiseopt_alloc (void)
{
	pathwiseopt	*path = (pathwiseopt *) malloc (sizeof (pathwiseopt));
	path->output_fullpath = false;
	path->output_bic_info = false;
	path->gamma_bic = 0.;
	path->log10_lambda1_lower = 0.;
	path->dlog10_lambda1 = 0.;
	path->beta_opt = NULL;
	path->lambda1_opt = 0.;
	path->nrm1_opt = 0.;
	path->min_bic_val = CDESCENT_POSINF;
	return path;
}

/*** create new pathwise optimization object ***/
pathwiseopt *
pathwiseopt_new (const double log10_lambda1_lower, const double dlog10_lambda1)
{
	pathwiseopt	*path = pathwiseopt_alloc ();
	strcpy (path->fn_path, default_fn_path);	// default filename
	strcpy (path->fn_bic, default_fn_bic);	// default filename
	path->log10_lambda1_lower = log10_lambda1_lower;
	path->dlog10_lambda1 = dlog10_lambda1;
	return path;
}

/*** free pathwise optimization object ***/
void
pathwiseopt_free (pathwiseopt *path)
{
	if (path) {
		if (path->beta_opt) mm_real_free (path->beta_opt);
		free (path);
	}
	return;
}

/*** set outputs_fullpath = true and specify output filename ***/
void
pathwiseopt_set_to_outputs_fullpath (pathwiseopt *path, const char *fn)
{
	path->output_fullpath = true;
	if (fn) strcpy (path->fn_path, fn);
	return;
}

/*** set outputs_bic_info = true and specify output filename ***/
void
pathwiseopt_set_to_outputs_bic_info (pathwiseopt *path, const char *fn)
{
	path->output_bic_info = true;
	if (fn) strcpy (path->fn_bic, fn);
	return;
}

/*** set gamma of eBIC ***/
void
pathwiseopt_set_gamma_bic (pathwiseopt *path, const double gamma_bic)
{
	path->gamma_bic = gamma_bic;
	return;
}
