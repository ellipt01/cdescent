/*
 * pathwise.h
 *
 *  Created on: 2014/11/03
 *      Author: utsugi
 */

#ifndef PATHWISE_H_
#define PATHWISE_H_

#include <stdbool.h>

typedef struct s_pathwise pathwise;

struct s_pathwise {

	char		fn_path[80];			// filename to output solution path
	bool		output_fullpath;		// whether output full solution path
	char		fn_bic[80];			// filename to output BIC info
	bool		output_bic_info;		// whether output BIC info
	double		gamma_bic;				// gamma for eBIC

	double		log10_lambda1_lower;	// lower bound of lambda1 on log10 scale
	double		dlog10_lambda1;		// increment of lambda1 on log10 scale

	double		min_bic_val;			// minimum BIC
	mm_dense	*beta_opt;				// optimal beta corresponding to min_bic_val
	double		lambda1_opt;			// optimal lambda1
	double		nrm1_opt;				// | bic_opt |

};

#endif /* PATHWISE_H_ */
