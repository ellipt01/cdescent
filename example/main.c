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

#include <cdescent.h>
#include "example.h"

char		fn[80] = "\0";
int			skipheaders = 0;
double		lambda2 = 0.;
double		start = 0.;
double		stop = 100.;
double		dt = 0.1;
double		gamma_bic = 0.;	// traditional BIC
int			maxiter = 100;

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

bool
read_params (int argc, char **argv)
{
	int		i;
	bool	status = true;
	double	_start = start;
	double	_dt = dt;
	double	_stop = stop;
	double	_gamma = gamma_bic;

	if (argc <= 1) return false;

	for (i = 1; i < argc; i++) {
		char	*p;

		if (argv[i][0] == '-') {

			switch (argv[i][1]) {

				case 'f':
					p = strrchr (argv[++i], ':');
					if (p) {
						strcpy (fn, argv[i]);
						fn[strlen (argv[i]) - strlen (p)] = '\0';
						skipheaders = atoi (++p);
					} else strcpy (fn, argv[i]);
				break;

				case 'l':
					lambda2 = (double) atof (argv[++i]);
				break;

				case 't':
					sscanf (argv[++i], "%lf:%lf:%lf", &_start, &_dt, &_stop);
				break;

				case 'g':
					_gamma = (double) atof (argv[++i]);
				break;

				case 'm':
					maxiter = atoi (argv[++i]);
				break;

				default:
				break;
			}
		}
	}
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
	fprintf (stream, "read file: \t\"%s\" (skip headers = %d)\n", fn, (int) skipheaders);
	fprintf (stream, "lambda1 :\t[%.2f : %.2f : %.2f]\n", start, dt, stop);
	fprintf (stream, "lambda2 :\t%.2f\n", lambda2);
	fprintf (stream, "maxiter :\t%d\n", maxiter);
	fprintf (stream, "\n###########################################################\n");
	return;
}

mm_mtx *
mm_mtx_real_penalty_smooth_sparse (const int n)
{
	int		i, j, k;
	int		nz = 2 * (n - 1);

	mm_mtx	*d = mm_mtx_real_new (MM_MTX_SPARSE, MM_MTX_UNSYMMETRIC, n - 1, n, nz);
	d->i = (int *) malloc (nz * sizeof (int));
	d->j = (int *) malloc (nz * sizeof (int));
	d->p = (int *) malloc ((n + 1) * sizeof (int));
	d->data = (double *) malloc (nz * sizeof (double));

	k = 0;
	d->p[0] = 0;
	for (j = 0; j < n; j++) {
		if (j > 0) {
			d->i[k] = j - 1;
			d->j[k] = j;
			d->data[k++] = -1.;
		}
		if (j < n - 1) {
			d->i[k] = j;
			d->j[k] = j;
			d->data[k++] = 1.;
		}
		d->p[j + 1] = k;
	}
	return d;
}

mm_mtx *
mm_mtx_real_penalty_smooth_dense (const int n)
{
	int		j;
	int		nz = (n - 1) * n;

	mm_mtx	*d = mm_mtx_real_new (MM_MTX_DENSE, MM_MTX_UNSYMMETRIC, n - 1, n, nz);
	d->data = (double *) malloc (nz * sizeof (double));

	for (j = 0; j < n - 1; j++) {
		d->data[j + j * d->m] = 1.;
		d->data[j + (j + 1) * d->m] = -1.;
	}
	return d;
}

mm_mtx *
create_mm_mtx_real_sparse (int m, int n, double *data)
{
	int		i, j, k = 0;
	mm_mtx	*x = mm_mtx_real_new (MM_MTX_SPARSE, MM_MTX_UNSYMMETRIC, m, n, m * n);
	x->i = (int *) malloc (x->nz * sizeof (int));
	x->j = (int *) malloc (x->nz * sizeof (int));
	x->p = (int *) malloc ((n + 1) * sizeof (int));
	x->data = data;

	x->p[0] = 0;
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			x->i[k] = i;
			x->j[k] = j;
			k++;
		}
		x->p[j + 1] = k;
	}
	return x;
}

mm_mtx *
create_mm_mtx_real_dense (int m, int n, double *data)
{
	int		k;
	mm_mtx	*x = mm_mtx_real_new (MM_MTX_DENSE, MM_MTX_UNSYMMETRIC, m, n, m * n);
	x->data = data;
	return x;
}

#include <time.h>

int
main (int argc, char **argv)
{
	int			m;
	int			n;
	double		*datax;
	double		*datay;
	linreg		*lreg;

	mm_mtx		*x;
	mm_mtx		*y;
	mm_mtx		*d;

	if (!read_params (argc, argv)) usage (argv[0]);
	fprintf_params (stderr);

	/* linear system */
	read_data (fn, skipheaders, &m, &n, &datay, &datax);
	y = create_mm_mtx_real_dense (m, 1, datay);
	x = create_mm_mtx_real_sparse (m, n, datax);
//	d = NULL;
//	d = mm_mtx_real_eye (n);
	d = mm_mtx_real_penalty_smooth_dense (n);
//	d = mm_mtx_real_penalty_smooth_sparse (n);

	lreg = linreg_alloc (y, x, lambda2, d);
	linreg_centering_y (lreg);
	linreg_centering_x (lreg);
	linreg_normalizing_x (lreg);

	{
		clock_t	t1, t2;
		t1 = clock ();
		example_cdescent_pathwise (lreg, start, dt, stop, 1.e-3, 1000);
		t2 = clock ();
		fprintf (stderr, "time = %.2e\n", (double) (t2 - t1) / CLOCKS_PER_SEC);
	}

	linreg_free (lreg);
	mm_mtx_free (x);
	mm_mtx_free (y);
	if (d) mm_mtx_free (d);

	return EXIT_SUCCESS;
}
