#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <mm_mtx.h>
#include "linreg_private.h"

mm_mtx *
mm_mtx_real_alloc (void)
{
	mm_mtx	*mm = (mm_mtx *) malloc (sizeof (mm_mtx));
	mm->m = 0;
	mm->n = 0;
	mm->nz = 0;
	mm->i = NULL;
	mm->j = NULL;
	mm->p = NULL;
	mm->data = NULL;

	// typdecode[3] = 'G'
	mm_initialize_typecode (&mm->typecode);
	// typecode[2] = 'R'
	mm_set_real (&mm->typecode);

	return mm;
}

mm_mtx *
mm_mtx_real_new (bool sparse, bool symmetric, const int m, const int n, const int nz)
{
	mm_mtx	*mm = mm_mtx_real_alloc ();

	mm->m = m;
	mm->n = n;
	mm->nz = nz;

	// typecode[3] = 'G'
	mm_initialize_typecode (&mm->typecode);

	// typecode[0] = 'M'
	mm_set_matrix (&mm->typecode);

	// typecode[1] = 'C' or 'A'
	if (sparse) mm_set_coordinate (&mm->typecode);
	else mm_set_array (&mm->typecode);

	// typecode[2] = 'R'
	mm_set_real (&mm->typecode);

	// typecode[3] = 'S'
	if (symmetric) mm_set_symmetric (&mm->typecode);

	return mm;
}

void
mm_array_set_all (int n, double *data, const double val)
{
	while (n-- >= 0) *data++ = val;
}

void
mm_mtx_real_set_all (mm_mtx *mm, const double val)
{
	mm_array_set_all (mm->nz, mm->data, val);
}

mm_mtx *
mm_mtx_real_eye (const int n)
{
	int		k;
	mm_mtx	*mm = mm_mtx_real_new (true, true, n, n, n);

	mm->i = (int *) malloc (n * sizeof (int));
	mm->j = (int *) malloc (n * sizeof (int));
	mm->data = (double *) malloc (n * sizeof (double));
	mm->p = (int *) malloc ((n + 1) * sizeof (int));

	mm->p[0] = 0;
	for (k = 0; k < n; k++) {
		mm->i[k] = k;
		mm->j[k] = k;
		mm->data[k] = 1.;
		mm->p[k + 1] = k + 1;
	}

	return mm;
}

mm_mtx *
mm_mtx_real_s_dot_d (bool trans, const double alpha, const mm_mtx *s, const mm_mtx *d, const double beta)
{
	int		j, k, l;
	int		m, n, lda;
	bool	symmetric;
	mm_mtx	*mm;

	if (!mm_is_sparse (s->typecode))
		linreg_error ("mm_mtx_real_s_dot_d", "matrix *s must be sparse.", __FILE__, __LINE__);
	if (!mm_is_dense (d->typecode))
		linreg_error ("mm_mtx_real_s_dot_d", "matrix *d must be dense.", __FILE__, __LINE__);

	m = (trans) ? s->n : s->m;
	n = (trans) ? s->m : s->n;
	if (n != d->m) linreg_error ("mm_mtx_real_s_dot_d", "matrix size mismatch.", __FILE__, __LINE__);

	lda = (trans) ? s->n : s->m;

	mm = mm_mtx_real_new (false, false, m, d->n, m * d->n);

	symmetric = mm_is_symmetric (s->typecode);

	mm->data = (double *) malloc (mm->nz * sizeof (double));
	mm_mtx_real_set_all (mm, beta);

	for (l = 0; l < d->n; l++) {
		for (j = 0; j < s->n; j++) {
			for (k = s->p[j]; k < s->p[j + 1]; k++) {
				int		si = (trans) ? s->i[k] : s->j[k];
				int		sj = (trans) ? s->j[k] : s->i[k];
				double	*ml = mm->data + l * lda;
				double	*dl = d->data + l * d->m;
				*(ml + sj) += alpha * s->data[k] * (*(dl + si));
				if (symmetric && j != sj) *(ml + si) += alpha * s->data[k] * (*(dl + sj));
			}
		}
	}

	return mm;
}

double
mm_mtx_real_sj_dot_d (const int j, const mm_mtx *s, const mm_mtx *d)
{
	int		k;
	double	val;

	if (!mm_is_sparse (s->typecode))
		linreg_error ("mm_mtx_real_sj_dot_d", "matrix *s must be sparse.", __FILE__, __LINE__);
	if (!mm_is_dense (d->typecode))
		linreg_error ("mm_mtx_real_sj_dot_d", "matrix *d must be dense.", __FILE__, __LINE__);

	val = 0;
	for (k = s->p[j]; k < s->p[j + 1]; k++) val += s->data[k] * d->data[s->i[k]];
	return val;
}

double
mm_mtx_real_sj_nrm (const int j, const mm_mtx *s)
{
	int		size;
	if (!mm_is_sparse (s->typecode))
		linreg_error ("mm_mtx_real_sj_nrm", "matrix *s must be sparse.", __FILE__, __LINE__);
	size = s->p[j + 1] - s->p[j];
	return dnrm2_ (&size, s->data + s->p[j], &ione);
}

mm_mtx *
mm_mtx_real_x_dot_y (bool trans, const double alpha, const mm_mtx *x, const mm_mtx *y, const double beta)
{
	int		m = (trans) ? x->n : x->m;
	int		n = (trans) ? x->m : x->n;
	mm_mtx	*c;

	if (n != y->m) linreg_error ("mm_mtx_real_x_dot_y", "matrix size mismatch.", __FILE__, __LINE__);

	c = mm_mtx_real_new (false, false, m, y->n, m * y->n);
	if (mm_is_sparse (x->typecode)) c = mm_mtx_real_s_dot_d (trans, alpha, x, y, beta);
	else {
		c->data = (double *) malloc (m * y->n * sizeof (double));
		dgemv_ ((trans) ? "T" : "N", &m, &n, &alpha, x->data, &y->m, y->data, &ione, &beta, c->data, &ione);
	}
	return c;
}

double
mm_mtx_real_xj_dot_y (const int j, const mm_mtx *x, const mm_mtx *y)
{
	double	val;

	if (j < 0 || x->n <= j) linreg_error ("mm_mtx_real_xj_dot_y", "index out of range.", __FILE__, __LINE__);

	if (mm_is_sparse (x->typecode)) val = mm_mtx_real_sj_dot_d (j, x, y);
	else val = ddot_ (&x->m, x->data + j * x->m, &ione, y->data, &ione);

	return val;
}

double
mm_mtx_real_xj_dot_xj (const int j, const mm_mtx *x)
{
	double	val;

	if (j < 0 || x->n <= j) linreg_error ("mm_mtx_real_xj_dot_xj", "index out of range.", __FILE__, __LINE__);

	if (mm_is_sparse (x->typecode)) val = mm_mtx_real_sj_nrm (j, x);
	else val = dnrm2_ (&x->m, x->data + j * x->m, &ione);

	return pow (val, 2.);
}
