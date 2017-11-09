/*
 * cdescent.h
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#ifndef CDESCENT_H_
#define CDESCENT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <mmreal.h>
#include <objects.h>
#include <linregmodel.h>
#include <regression.h>

/* cdescent.c */
cdescent	*cdescent_new (const double alpha, const linregmodel *lreg, const double tol, const int maxiter, bool parallel);
void		cdescent_free (cdescent *cd);

void		cdescent_set_cyclic (cdescent *cd);
void		cdescent_set_stochastic (cdescent *cd, const unsigned int *seed);

bool		cdescent_set_penalty_factor (cdescent *cd, const mm_dense *w, const double tau);

void		cdescent_not_use_intercept (cdescent *cd);
void		cdescent_set_constraint (cdescent *cd, constraint_func func);
void		cdescent_use_fixed_lambda (cdescent *cd, const double lambda);

void		cdescent_set_lambda (cdescent *cd, const double lambda);
void		cdescent_set_log10_lambda (cdescent *cd, const double log10_lambda);

void		cdescent_set_log10_lambda_upper (cdescent *cd, const double log10_lambda_upper);
void		cdescent_set_log10_lambda_lower (cdescent *cd, const double log10_lambda_lower);
void		cdescent_set_log10_dlambda (cdescent *cd, const double dlog10_lambda);
void		cdescent_set_outputs_fullpath (cdescent *cd, const char *fn);
void		cdescent_set_outputs_info (cdescent *cd, const char *fn);

double		cdescent_get_intercept_in_original_scale (const cdescent *cd);
mm_dense	*cdescent_get_beta_in_original_scale (const cdescent *cd);
void		cdescent_init_beta (cdescent *cd, const mm_dense *beta);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
