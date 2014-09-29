/*
 * bic.h
 *
 *  Created on: 2014/09/29
 *      Author: utsugi
 */

#ifndef BIC_H_
#define BIC_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Extended Bayesian Information Criterion (Chen and Chen, 2008)
 * eBIC = log(rss) + df * (log(m) + 2 * gamma * log(n)) / m
 * 	if gamma = 0, eBIC is identical with the classical BIC */
typedef struct s_bic_info	bic_info;

struct s_bic_info
{
	double	m;		// number of data
	double	n;		// number of variables
	double	rss;	// residual sum of squares
	double	df;		// degree of freedom
	double	gamma;	// tuning parameter for eBIC
	double	bic_val;	// value of eBIC
};

#ifdef __cplusplus
}
#endif

#endif /* BIC_H_ */
