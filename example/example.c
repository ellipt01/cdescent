/*
 * example.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cdescent.h>

/*** 1D derivation operator for the L2 penalty of s-lasso ***/

static mm_sparse *
penalty_ssmooth (const int n)
{
	int			j, k;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n - 1, n, 2 * (n - 1));

	k = 0;
	for (j = 0; j < n; j++) {
		if (j > 0) {
			s->i[k] = j - 1;
			s->data[k++] = -1.;
		}
		if (j < n - 1) {
			s->i[k] = j;
			s->data[k++] = 1.;
		}
		s->p[j + 1] = k;
	}
	return s;
}

static mm_dense *
penalty_dsmooth (const int n)
{
	int		j;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n - 1, n, (n - 1) * n);
	mm_real_set_all (d, 0.);
	for (j = 0; j < n; j++) {
		if (j > 0) d->data[j - 1 + j * d->m] = -1.;
		if (j < n - 1) d->data[j + j * d->m] = 1.;
	}
	return d;
}

/*** sparse/dense 1D derivation operator ***/
mm_real *
penalty_smooth (MMRealFormat format, const int n)
{
	return (format == MM_REAL_SPARSE) ? penalty_ssmooth (n) : penalty_dsmooth (n);
}

extern char		*optarg;
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
	fprintf (stderr, "              -r <log10_lambda1_min:d_log10_lambda1; default = -2:0.1>\n");
	fprintf (stderr, "              -g <gamma of eBIC in [0, 1]; default = 0>\n");
	fprintf (stderr, "              -t <tolerance; default = 1.e-3> }\n\n");
	fprintf (stderr, "              -m <maxiters; default = 100000> }\n\n");
	exit (1);
}

/*** parameters ***/
extern char		infn_x[];
extern char		infn_y[];
extern double	alpha;
extern double	log10_lambda;
extern double	dlog10_lambda;
extern double	gamma_bic;	// classical BIC
extern double	tolerance;
extern int		maxiter;

/*** read command line options ***/
bool
read_params (int argc, char **argv)
{
	bool	status = true;
	char	c;

	while ((c = getopt (argc, argv, "x:y:a:r:g:t:m:")) != -1) {

		switch (c) {

			case 'x':
				strcpy (infn_x, optarg);
				break;

			case 'y':
				strcpy (infn_y, optarg);
				break;

			case 'a':
					alpha = (double) atof (optarg);
				break;

			case 'r':
					if (strchr (optarg, ':')) {
						sscanf (optarg, "%lf:%lf", &log10_lambda, &dlog10_lambda);
					} else log10_lambda = (double) atof (optarg);
				break;

			case 'g':
					gamma_bic = (double) atof (optarg);
				break;

			case 't':
					tolerance = (double) atof (optarg);
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
