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
#include <omp.h>

#include <cdescent.h>
#include "example.h"

char		fn[80] = "\0";
int			skipheaders = 0;
double		lambda2 = 0.;
double		start = 0.;
double		stop = 100.;
double		dt = 0.1;
double		gamma_bic = 0.;	// traditional BIC
int			maxiter = 100000;

void
usage (char *toolname)
{
	char	*p = strrchr (toolname, '/');
	if (p) p++;
	else p = toolname;

	fprintf (stderr, "\nUSAGE:\n%s -f <input_file>{:skipheaders} -l <lambda2> \n", p);
	fprintf (stderr, "[optional] { -t <start>:<dt>:<stop> -g <gamma of EBIC in [0, 1]> -m <maxsteps> }\n\n");
	exit (1);
}

extern char	*optarg;

bool
read_params (int argc, char **argv)
{
	int		i;
	bool	status = true;
	double	_start = start;
	double	_dt = dt;
	double	_stop = stop;
	double	_gamma = gamma_bic;
	char	c;

	while ((c = getopt (argc, argv, "f:l:t:g:m:")) != -1) {
		char *p;

		switch (c) {

			case 'f':
				p = strrchr (optarg, ':');
				if (p) {
					strcpy (fn, optarg);
					fn[strlen (optarg) - strlen (p)] = '\0';
					skipheaders = atoi (++p);
				} else strcpy (fn, optarg);
				break;

			case 'l':
					lambda2 = (double) atof (optarg);
				break;

			case 't':
					sscanf (optarg, "%lf:%lf:%lf", &_start, &_dt, &_stop);
				break;

			case 'g':
					_gamma = (double) atof (optarg);
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

	if (strlen (fn) <= 1) {
		fprintf (stderr, "ERROR: input file name is not specified.\n");
		status = false;
	}
	if (_start >= _stop || floor ((_stop - _start) / _dt) <= 0) {
		fprintf (stderr, "ERROR: range of lambda1 invalid : %.2f:%.2f:%.2f\n", _start, _dt, _stop);
		status = false;
	}
	if (_gamma < 0. || 1. < _gamma) {
		fprintf (stderr, "ERROR: gamma (%f) must be [0, 1].\n", _gamma);
		status = false;
	}

	start = _start;
	dt = _dt;
	stop = _stop;
	gamma_bic = _gamma;

	return status;
}

void
fprintf_params (FILE *stream)
{
	fprintf (stream, "###########################################################\n\n");
	fprintf (stream, "read file:\t\"%s\" (skip headers = %d)\n", fn, (int) skipheaders);
	fprintf (stream, "log10(lambda1):\t[%.2f : %.2f : %.2f]\n", start, dt, stop);
	fprintf (stream, "lambda2:\t%.2f\n", lambda2);
	fprintf (stream, "maxiter:\t%d\n", maxiter);
	fprintf (stream, "\n###########################################################\n");
	return;
}

static mm_sparse *
mm_real_penalty_ssmooth (const int n)
{
	int		i, j, k;
	int		nz = 2 * (n - 1);

	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_UNSYMMETRIC, n - 1, n, nz);
	s->i = (int *) malloc (nz * sizeof (int));
	s->p = (int *) malloc ((n + 1) * sizeof (int));
	s->data = (double *) malloc (nz * sizeof (double));

	k = 0;
	s->p[0] = 0;
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
mm_real_penalty_dsmooth (const int n)
{
	int			j;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_UNSYMMETRIC, n - 1, n, (n - 1) * n);
	d->data = (double *) malloc (d->nz * sizeof (double));
	for (j = 0; j < n; j++) {
		if (j > 0) d->data[j + (j + 1) * d->m] = -1.;
		if (j < n - 1) d->data[j + j * d->m] = 1.;
	}
	return d;
}

mm_real *
mm_real_penalty_smooth (MMRealFormat format, const int n)
{
	return (format == MM_REAL_SPARSE) ? mm_real_penalty_ssmooth (n) : mm_real_penalty_dsmooth (n);
}

mm_sparse *
create_mm_sparse (int m, int n, double *data, double threshold)
{
	int			j, k, l;
	mm_sparse	*x = mm_real_new (MM_REAL_SPARSE, MM_REAL_UNSYMMETRIC, m, n, m * n);
	x->i = (int *) malloc (x->nz * sizeof (int));
	x->p = (int *) malloc ((n + 1) * sizeof (int));
	x->data = (double *) malloc (x->nz * sizeof (double));

	k = 0;
	l = 0;
	x->p[0] = 0;
	for (j = 0; j < n; j++) {
		int		i;
		for (i = 0; i < m; i++) {
			if (fabs (data[l++]) > threshold) {
				x->i[k] = i;
				x->data[k] = data[k];
				k++;
			}
		}
		x->p[j + 1] = k;
	}
	fprintf (stderr, "x->nz = %d -> ", x->nz);
	mm_real_realloc (x, k);
	fprintf (stderr, "x->nz = %d\n", x->nz);
	return x;
}

mm_dense *
create_mm_dense (int m, int n, double *data)
{
	int			k;
	mm_dense	*x = mm_real_new (MM_REAL_DENSE, MM_REAL_UNSYMMETRIC, m, n, m * n);
	x->data = (double *) malloc (x->nz * sizeof (double));
	for (k = 0; k < m * n; k++) x->data[k] = data[k];
	return x;
}

int
main (int argc, char **argv)
{
	linregmodel	*lreg;

	mm_real		*x;
	mm_dense		*y;
	mm_real		*d;

	if (!read_params (argc, argv)) usage (argv[0]);
	fprintf_params (stderr);

	/* linear system */
	{
		int		m;
		int		n;
		double	*datax;
		double	*datay;
		read_data (fn, skipheaders, &m, &n, &datay, &datax);
		y = create_mm_dense (m, 1, datay);
		free (datay);
		x = create_mm_dense (m, n, datax);
		free (datax);
	}
//	d = NULL;
//	d = mm_real_eye (MM_REAL_DENSE, x->n);
	d = mm_real_eye (MM_REAL_SPARSE, x->n);
//	d = mm_real_penalty_smooth (MM_REAL_DENSE, x->n);
//	d = mm_real_penalty_smooth (MM_REAL_SPARSE, x->n);

	lreg = linregmodel_new (y, x, lambda2, d, false, true, true, true);

	{
#ifdef _OPENMP
		double	t1, t2;
		t1 = omp_get_wtime ();
#endif
		example_cdescent_pathwise (lreg, start, dt, stop, 1.e-3, maxiter, true);
#ifdef _OPENMP
		t2 = omp_get_wtime ();
		fprintf (stderr, "time = %.2e\n", t2 - t1);
#endif
	}

	linregmodel_free (lreg);
	mm_real_free (x);
	mm_real_free (y);
	if (d) mm_real_free (d);

	return EXIT_SUCCESS;
}
