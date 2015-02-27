/*
 * pathwiseopt.h
 *
 *  Created on: 2014/11/03
 *      Author: utsugi
 */

#ifndef PATHWISEOPT_H_
#define PATHWISEOPT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <s_cdescent.h>

/* weighting function */
typedef mm_dense* (*weight_func) (int i, cdescent *cd, void *data);

/* reweighting function
 * the function weighting_func	function is called to calculate weight
 * for each pathwiseopt iteration */
typedef struct s_reweighting_func reweighting_func;

struct s_reweighting_func {
	double		tau;
	weight_func	function;
	void		*data;
};

/*** object of pathwise coordinate optimization ***/
typedef struct s_pathwiseopt pathwiseopt;

struct s_pathwiseopt {

	bool		was_modified;		// whether this object was modified after initialization

	char		fn_path[80];		// file to output solution path
	bool		output_fullpath;	// whether to outputs full solution path
	char		fn_bic[80];			// file to output BIC info
	bool		output_bic_info;	// whether to outputs BIC info
	double		gamma_bic;			// gamma for eBIC

	double		log10_lambda1_lower;// lower bound of lambda1 on log10 scale
	double		dlog10_lambda1;		// increment of lambda1 on log10 scale

	double		min_bic_val;		// minimum BIC
	mm_dense	*beta_opt;			// optimal beta corresponding to min_bic_val
	double		lambda1_opt;		// optimal lambda1
	double		nrm1_opt;			// | beta_opt |

	reweighting_func	*func;		// reweighting function

};

reweighting_func * reweighting_function_new (const double tau, const weight_func func, void *data);

pathwiseopt	*pathwiseopt_new (const double log10_lambda1_lower, const double dlog10_lambda1);
void		pathwiseopt_free (pathwiseopt *path);
void		pathwiseopt_set_to_outputs_fullpath (pathwiseopt *path, const char *fn);
void		pathwiseopt_set_to_outputs_bic_info (pathwiseopt *path, const char *fn);
void		pathwiseopt_set_gamma_bic (pathwiseopt *path, const double gamma_bic);
void		pathwiseopt_set_reweighting (pathwiseopt *path, reweighting_func *func);

#ifdef __cplusplus
}
#endif

#endif /* PATHWISEOPT_H_ */
