/*
 * settings.h
 *
 *  Created on: 2015/04/12
 *      Author: utsugi
 */

#ifndef SETTINGS_H_
#define SETTINGS_H_

/*** default settings ***/
char			infn_x[80] = "\0";		// store input file name of design matrix
char			infn_y[80] = "\0";		// store input file name of observed data
double			alpha = 0.;			// L2 penalty parameter
double			log10_lambda = -2.;		// start log10(lambda1) for warm start
double			dlog10_lambda = 0.1;	// increment of log10(lambda1)
double			gamma_bic = 0.;			// classical BIC

double			tolerance = 1.e-3;
int				maxiter = 100000;

#endif /* SETTINGS_H_ */
