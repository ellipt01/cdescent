/*
 * main.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include <cdescent.h>

#include "example.h"

/*** An example program of elastic net regression using cdescent library. ***/

char			infn_x[80];
char			infn_y[80];

/*** default settings ***/
double			lambda2 = 0.;
double			log10_lambda1 = -2.;
double			dlog10_lambda1 = 0.1;
double			gamma_bic = 0.;	// classical BIC

double			tolerance = 1.e-3;
int				maxiter = 100000;

extern char	*optarg;
extern int		optind;

/*** print usage ***/
void
usage (char *toolname)
{
	char	*p = strrchr (toolname, '/');
	if (p) p++;
	else p = toolname;
	fprintf (stderr, "\nUSAGE:\n%s -x <input file of matrix x> -y <input file of vector y>\n", p);
	fprintf (stderr, "[optional]  { -l <lambda2; default = 0>\n");
	fprintf (stderr, "              -t <log10_lambda1_min:d_log10_lambda1; default = -2:0.1>\n");
	fprintf (stderr, "              -g <gamma of eBIC in [0, 1]; default = 0>\n");
	fprintf (stderr, "              -m <maxiters; default = 100000> }\n\n");
	exit (1);
}

/*** read command line options ***/
bool
read_params (int argc, char **argv)
{
	bool	status = true;
	char	c;

	while ((c = getopt (argc, argv, "x:y:l:t:g:m:")) != -1) {

		switch (c) {

			case 'x':
				strcpy (infn_x, optarg);
				break;

			case 'y':
				strcpy (infn_y, optarg);
				break;

			case 'l':
					lambda2 = (double) atof (optarg);
				break;

			case 't':
					if (strchr (optarg, ':')) {
						sscanf (optarg, "%lf:%lf", &log10_lambda1, &dlog10_lambda1);
					} else log10_lambda1 = (double) atof (optarg);
				break;

			case 'g':
					gamma_bic = (double) atof (optarg);
				break;

			case 'm':
					maxiter = atoi (optarg);
				break;

			case ':':
					fprintf (stdout, "%c needs value.\n", c);
      				break;

			case '?':
      				fprintf (stdout, "unknown option\n");
      				break;

			default:
				break;
		}
	}
	for(; optind < argc; optind++) fprintf (stdout, "%s\n", argv[optind]);

	if (strlen (infn_x) <= 1 || strlen (infn_y) <= 1) {
		fprintf (stderr, "ERROR: input file name is not specified.\n");
		status = false;
	}

	return status;
}

mm_dense *
weight_func0 (int iter, cdescent *cd, void *data)
{
	int			i;
	int			m = cd->beta->m;
	mm_dense	*w = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, 1, m);
	double		eps = 1.e-3;
	for (i = 0; i < m; i++) w->data[i] = 1. / (fabs (cd->beta->data[i]) + eps);
	return w;
}

int
main (int argc, char **argv)
{
	mm_dense	*x;
	mm_dense	*y;
	mm_real		*d;

	linregmodel	*lreg;

	cdescent	*cd;

	pathwiseopt	*path;

	FILE		*fp;

	/*** read command line options ***/
	if (!read_params (argc, argv)) usage (argv[0]);

	/*** prepare observation, predictors and L2 penalty ***/
	/* read observation mm_dense *y from file */
	if ((fp = fopen (infn_y, "r")) == NULL) {
		fprintf (stderr, "ERROR: cannot open file %s.\n", infn_y);
		exit (1);
	}
	y = mm_real_fread (fp);
	fclose (fp);

	/* read predictors mm_real *x from file */
	if ((fp = fopen (infn_x, "r")) == NULL) {
		fprintf (stderr, "ERROR: cannot open file %s.\n", infn_x);
		exit (1);
	}
	x = mm_real_fread (fp);
	fclose (fp);

	/* L2 penalty */
	//	d = NULL;	// no L2 penalty for lasso

	// sparse identity matrix for elastic net
	d = mm_real_eye (MM_REAL_SPARSE, x->n);

	// sparse 1D derivation operator for s-lasso
	//	d = penalty_smooth (MM_REAL_SPARSE, x->n);	// see example.c

	/*** create linear regression model object
	     for || y - x * beta ||^2 + lambda2 * || d * beta ||^2 ***/
	lreg = linregmodel_new (y, x, lambda2, d, DO_CENTERING_Y | DO_STANDARDIZING_X);

	mm_real_free (y);
	mm_real_free (x);
	if (d) mm_real_free (d);

	/*** create CCD (cyclic coordinate descent) object ***/
	cd = cdescent_new (lreg, tolerance, maxiter, false);

	/*** create pathwise CCD optimization object ***/
	path = pathwiseopt_new (log10_lambda1, dlog10_lambda1);
	pathwiseopt_set_to_outputs_fullpath (path, NULL);	// output full solution path
	pathwiseopt_set_to_outputs_bic_info (path, NULL);	// output BIC info
	pathwiseopt_set_gamma_bic (path, gamma_bic);		// set gamma for eBIC
	{
		reweighting_func	*func = reweighting_function_new (1., weight_func0, NULL);
		pathwiseopt_set_reweighting (path, func);
	}

	/*** do pathwise CCD regression ***/
	cdescent_cyclic_pathwise (cd, path);

	fprintf (stderr, "lambda1_opt = %.2f, nrm1(beta_opt) = %.2f, min_bic = %.2f\n",
		path->lambda1_opt, path->nrm1_opt, path->min_bic_val);

	pathwiseopt_free (path);
	cdescent_free (cd);
	linregmodel_free (lreg);

	return EXIT_SUCCESS;
}
