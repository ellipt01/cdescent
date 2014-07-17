/*
 * example.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "example.h"

extern int		skipheaders;
extern double	lambda2;
extern double	start;
extern double	stop;
extern double	dt;
extern double	gamma_bic;
extern int		maxiter;

extern char	*optarg;

void
usage (char *toolname)
{
	char	*p = strrchr (toolname, '/');
	if (p) p++;
	else p = toolname;

	fprintf (stderr, "\nUSAGE:\n%s -f <input_file>{:num skipheaders} -l <lambda2> \n", p);
	fprintf (stderr, "[optional] { -t <log10_lambda1_min>:<d_log10_lambda1>:<log10_lambda1_max>\n");
	fprintf (stderr, "             -g <gamma of EBIC in [0, 1]> -m <maxsteps> }\n\n");
	exit (1);
}

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
					strcpy (infn, optarg);
					infn[strlen (optarg) - strlen (p)] = '\0';
					skipheaders = atoi (++p);
				} else strcpy (infn, optarg);
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

	if (strlen (infn) <= 1) {
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
	fprintf (stream, "read file:\t\"%s\" (skip headers = %d)\n", infn, skipheaders);
	fprintf (stream, "log10(lambda1):\t[%.2f : %.2f : %.2f]\n", start, dt, stop);
	fprintf (stream, "lambda2:\t%.2f\n", lambda2);
	fprintf (stream, "maxiter:\t%d\n", maxiter);
	fprintf (stream, "\n###########################################################\n");
	return;
}

static void
count_data (char *fn, int skip_header, int *row, int *col)
{
	int		ndata = 0;
	int		npred = 0;
	char	buf[100 * BUFSIZ];
	FILE	*fp;

	*row = 0;
	*col = 0;

	if ((fp = fopen (fn, "r")) == NULL) {
		fprintf (stderr, "ERROR: cannot open file %s.\n", fn);
		exit (1);
	}
	while (fgets (buf, 100 * BUFSIZ, fp) != NULL) {
		if (buf[0] == '#' || buf[0] == '\n') continue;
		if (ndata - skip_header == 0) {
			char	*p;
			for (p = strtok (buf, "\t "); p != NULL; p = strtok (NULL, "\t ")) npred++;
		}
		ndata++;
	}
	fclose (fp);
	*row = ndata - skip_header;
	*col = npred - 1;
	return;
}

void
read_data (char *fn, int skip_header, int *n, int *p, double **y, double **x)
{
	int		i, j;
	int		size1;
	int		size2;

	char	buf[100 * BUFSIZ];
	FILE	*fp;

	double	*_y;
	double	*_x;

	count_data (fn, skip_header, &size1, &size2);

	_y = (double *) malloc (size1 * sizeof (double));
	_x = (double *) malloc (size1 * size2 * sizeof (double));

	if ((fp = fopen (fn, "r")) == NULL) return;
	i = 0;
	while (fgets (buf, 100 * BUFSIZ, fp) != NULL) {
		char	*p;
		if (buf[0] == '#' || buf[0] == '\n') continue;
		if (i - skip_header >= 0) {
			for (j = 0, p = strtok (buf, "\t "); p != NULL; j++, p = strtok (NULL, "\t ")) {
				double	val = (double) atof (p);
				if (j >= size2) _y[i - skip_header] = val;
				else _x[i - skip_header + j * size1] = val;
			}
		}
		i++;
	}
	fclose (fp);

	*n = size1;
	*p = size2;
	*x = _x;
	*y = _y;

	return;
}

mm_dense *
create_mm_dense (int m, int n, double *data)
{
	int			k;
	mm_dense	*x = mm_real_new (MM_REAL_DENSE, false, m, n, m * n);
	x->data = (double *) malloc (x->nz * sizeof (double));
	for (k = 0; k < m * n; k++) x->data[k] = data[k];
	return x;
}

static mm_sparse *
mm_real_penalty_ssmooth (const int n)
{
	int		i, j, k;
	int		nz = 2 * (n - 1);

	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, false, n - 1, n, nz);
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
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, false, n - 1, n, (n - 1) * n);
	d->data = (double *) malloc (d->nz * sizeof (double));
	mm_real_set_all (d, 0.);
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

