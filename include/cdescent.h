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
#include <bic.h>

/* cdescent.c */
cdescent	*cdescent_new (const linregmodel *lreg, const double tol, const int maxiter, bool parallel);
void		cdescent_free (cdescent *cd);
bool		cdescent_set_penalty_factor (cdescent *cd, const mm_dense *w, const double tau);
bool		cdescent_set_lambda1 (cdescent *cd, const double lambda1);
bool		cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1);
void		cdescent_set_update_intercept (cdescent *cd, bool update_intercept);

void		cdescent_set_pathwise_log10_lambda1_upper (cdescent *cd, const double log10_lambda1_upper);
void		cdescent_set_pathwise_log10_lambda1_lower (cdescent *cd, const double log10_lambda1_lower);
void		cdescent_set_pathwise_dlog10_lambda1 (cdescent *cd, const double dlog10_lambda1);
void		cdescent_set_pathwise_outputs_fullpath (cdescent *cd, const char *fn);
void		cdescent_set_pathwise_outputs_bic_info (cdescent *cd, const char *fn);
void		cdescent_set_pathwise_gamma_bic (cdescent *cd, const double gamma_bic);

reweighting_func * reweighting_function_new (const double tau, const weight_func func, void *data);
void		cdescent_set_pathwise_reweighting (cdescent *cd, reweighting_func *func);

/* regression.c */
bool		cdescent_do_cyclic_update (cdescent *cd);
bool		cdescent_do_pathwise_optimization (cdescent *cd);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
