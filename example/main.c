/*
 * main.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "example.h"

/*** default settings ***/
double	lambda2 = 0.;
double	log10_lambda1 = -2.;
double	dlog10_lambda1 = 0.1;
double	gamma_bic = 0.;	// traditional BIC

double	tolerance = 1.e-3;
int		maxiter = 100000;

extern char	*optarg;

/*** print usage ***/
void
usage (char *toolname)
{
	char	*p = strrchr (toolname, '/');
	if (p) p++;
	else p = toolname;

	fprintf (stderr, "\nUSAGE:\n%s -x <input file of matrix x> -y <input file of vector y> -l <lambda2> \n", p);
	fprintf (stderr, "[optional] { -t <log10_lambda1_min>:<d_log10_lambda1>\n");
	fprintf (stderr, "             -g <gamma of EBIC in [0, 1]> -m <maxsteps> }\n\n");
	exit (1);
}

/*** read parameters ***/
bool
read_params (int argc, char **argv)
{
	bool	status = true;
	char	c;

	while ((c = getopt (argc, argv, "x:y:l:t:g:m:")) != -1) {
		char *p;

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
	if (gamma_bic < 0. || 1. < gamma_bic) {
		fprintf (stderr, "ERROR: gamma (%f) must be [0, 1].\n", gamma_bic);
		status = false;
	}

	return status;
}

/*** read infiles and create linregmodel ***/
linregmodel *
create_linregmodel (void)
{
	mm_dense	*x;
	mm_dense	*y;
	mm_real	*d;
	bool		has_copy = true;
	FILE		*fp;

	linregmodel	*lreg;

	if ((fp = fopen (infn_y, "r")) == NULL) {
		fprintf (stderr, "ERROR: cannot open file %s.\n", infn_y);
		exit (1);
	}
	y = mm_real_fread (fp);
	fclose (fp);

	if ((fp = fopen (infn_x, "r")) == NULL) {
		fprintf (stderr, "ERROR: cannot open file %s.\n", infn_x);
		exit (1);
	}
	x = mm_real_fread (fp);
	fclose (fp);

	// general penalty term
	//	d = NULL;	// lasso
	d = mm_real_eye (MM_REAL_SPARSE, x->n);	// elastic net
	//	d = mm_real_penalty_smooth (MM_REAL_SPARSE, x->n);	// s-lasso

	lreg = linregmodel_new (y, x, lambda2, d, has_copy, DO_CENTERING_Y | DO_STANDARDIZING_X);

	mm_real_free (x);
	mm_real_free (y);
	if (d) mm_real_free (d);

	return lreg;
}


int
main (int argc, char **argv)
{
	linregmodel	*lreg;
	cdescent		*cd;

	if (!read_params (argc, argv)) usage (argv[0]);

	/* create linear regression model object */
	lreg = create_linregmodel ();

	/* create cdescent object */
	cd = cdescent_new (lreg, tolerance, false);

	/* evaluate regression coefficients beta
	 * corresponding to the specified L1 regularization parameter log_lambda1 */
	{
#ifdef _OPENMP
		double	t1, t2;
		t1 = omp_get_wtime ();
#endif

		example_cdescent_pathwise (cd, log10_lambda1, dlog10_lambda1, maxiter, true);

#ifdef _OPENMP
		t2 = omp_get_wtime ();
		fprintf (stderr, "time = %.2e\n", t2 - t1);
#endif
	}

	/* output beta and mu corresponding to log_lambda1, in MatrixMarket format */
	{
		FILE	*fp;
		/* output resultant beta */
		if ((fp = fopen ("beta.res", "w"))) {
			mm_real_fwrite (fp, cd->beta, "%.4e");
			fclose (fp);
		}
		/* output mu (= X * beta) */
		if ((fp = fopen ("mu.res", "w"))) {
			mm_real_fwrite (fp, cd->mu, "%.4e");
			fclose (fp);
		}
	}

	cdescent_free (cd);
	linregmodel_free (lreg);

	return EXIT_SUCCESS;
}
