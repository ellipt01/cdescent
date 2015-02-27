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
#include <linregmodel.h>
#include <bic.h>
#include <s_cdescent.h>
#include <pathwiseopt.h>

/* cdescent.c */
cdescent	*cdescent_new (const linregmodel *lreg, const double tol, const int maxiter, bool parallel);
void		cdescent_free (cdescent *cd);
bool		cdescent_set_penalty_factor (cdescent *cd, const mm_dense *w, const double tau);
bool		cdescent_set_lambda1 (cdescent *cd, const double lambda1);
bool		cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1);

/* regression.c */
bool		cdescent_cyclic_update_once_cycle (cdescent *cd);
bool		cdescent_cyclic_update (cdescent *cd);
bool		cdescent_cyclic_pathwise (cdescent *cd, pathwiseopt *path);

/* bic.c */
bic_info	*cdescent_eval_bic (const cdescent *cd, double gamma);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
