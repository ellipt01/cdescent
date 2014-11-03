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

#include <stdbool.h>

/*** object of pathwise coordinate optimization ***/
typedef struct s_pathwiseopt pathwiseopt;

struct s_pathwiseopt {

	char		fn_path[80];			// file to output solution path
	bool		output_fullpath;		// whether to outputs full solution path
	char		fn_bic[80];			// file to output BIC info
	bool		output_bic_info;		// whether to outputs BIC info
	double		gamma_bic;				// gamma for eBIC

	double		log10_lambda1_lower;	// lower bound of lambda1 on log10 scale
	double		dlog10_lambda1;		// increment of lambda1 on log10 scale

	double		min_bic_val;			// minimum BIC
	mm_dense	*beta_opt;				// optimal beta corresponding to min_bic_val
	double		lambda1_opt;			// optimal lambda1
	double		nrm1_opt;				// | bic_opt |

};

pathwiseopt	*pathwiseopt_new (const double log10_lambda1_lower, const double dlog10_lambda1);
void			pathwiseopt_free (pathwiseopt *path);
void			pathwiseopt_set_output_fullpath (pathwiseopt *path, const char *fn);
void			pathwiseopt_set_output_bic_info (pathwiseopt *path, const char *fn);
void			pathwiseopt_set_gamma_bic (pathwiseopt *path, const double gamma_bic);

#ifdef __cplusplus
}
#endif

#endif /* PATHWISEOPT_H_ */
