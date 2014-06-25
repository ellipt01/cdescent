#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <mm_mtx.h>
#include "private.h"

mm_mtx *
mm_mtx_alloc (void)
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
mm_mtx_new (MM_MtxType type, MM_MtxSymmetric symmetric, const int m, const int n, const int nz)
{
	mm_mtx	*mm = mm_mtx_alloc ();

	mm->m = m;
	mm->n = n;
	mm->nz = nz;

	// typecode[3] = 'G'
	mm_initialize_typecode (&mm->typecode);

	// typecode[0] = 'M'
	mm_set_matrix (&mm->typecode);

	// typecode[1] = 'C' or 'A'
	if (type == MM_MTX_SPARSE) mm_set_coordinate (&mm->typecode);
	else mm_set_array (&mm->typecode);

	// typecode[2] = 'R'
	mm_set_real (&mm->typecode);

	// typecode[3] = 'S'
	if (symmetric == MM_MTX_SYMMETRIC) mm_set_symmetric (&mm->typecode);

	return mm;
}

void
mm_mtx_free (mm_mtx *mm)
{
	if (mm) {
		if (mm->i) free (mm->i);
		if (mm->j) free (mm->j);
		if (mm->p) free (mm->p);
		if (mm->data) free (mm->data);
		free (mm);
	}
	return;
}

bool
mm_mtx_realloc (mm_mtx *mm, const int nz)
{
	if (!mm) cdescent_error ("mm_mtxloc", "input matrix is empty.", __FILE__, __LINE__);
	if (mm->nz == nz) return true;
	mm->nz = nz;
	mm->data = (double *) realloc (mm->data, mm->nz * sizeof (double));
	if (mm->data == NULL) return false;
	if (mm_is_sparse (mm->typecode)) {
		mm->i = (int *) realloc (mm->i, mm->nz * sizeof (int));
		mm->j = (int *) realloc (mm->j, mm->nz * sizeof (int));
		if (mm->i == NULL || mm->j == NULL) return false;
	}
	return true;
}

static mm_mtx *
mm_mtx_copy_sparse (const mm_mtx *orig)
{
	int					k;
	MM_MtxSymmetric	symmetric = (mm_is_symmetric (orig->typecode)) ? MM_MTX_SYMMETRIC : MM_MTX_UNSYMMETRIC;
	mm_mtx				*dest = mm_mtx_new (MM_MTX_SPARSE, symmetric, orig->m, orig->n, orig->nz);

	dest->i = (int *) malloc (orig->nz * sizeof (int));
	dest->j = (int *) malloc (orig->nz * sizeof (int));
	dest->p = (int *) malloc ((orig->n + 1) * sizeof (int));
	dest->data = (double *) malloc (orig->nz * sizeof (double));

	for (k = 0; k < orig->nz; k++) {
		dest->i[k] = orig->i[k];
		dest->j[k] = orig->j[k];
		dest->data[k] = orig->data[k];
	}
	for (k = 0; k <= orig->n; k++)	dest->p[k] = orig->p[k];

	return dest;
}

static mm_mtx *
mm_mtx_copy_dense (const mm_mtx *orig)
{
	int		k;
	mm_mtx	*dest = mm_mtx_new (MM_MTX_DENSE, MM_MTX_UNSYMMETRIC, orig->m, orig->n, orig->nz);
	dest->data = (double *) malloc (orig->nz * sizeof (double));
	for (k = 0; k < orig->nz; k++) dest->data[k] = orig->data[k];
	return dest;
}

mm_mtx *
mm_mtx_copy (const mm_mtx *mm)
{
	if (!mm) cdescent_error ("mm_mtx_copy", "input matrix is empty.", __FILE__, __LINE__);
	return (mm_is_sparse (mm->typecode)) ? mm_mtx_copy_sparse (mm) : mm_mtx_copy_dense (mm);
}

void
mm_array_set_all (int n, double *data, const double val)
{
	while (n-- >= 0) *data++ = val;
}

void
mm_mtx_set_all (mm_mtx *mm, const double val)
{
	mm_array_set_all (mm->nz, mm->data, val);
}

mm_dense *
mm_mtx_sparse_to_dense (const mm_sparse *s)
{
	int			k;
	mm_dense	*d;
	if (!mm_is_sparse (s->typecode)) {
		cdescent_warning ("mm_sparse_to_dense", "input matrix is not sparse.", __FILE__, __LINE__);
		return mm_mtx_copy (s);
	}

	d = mm_mtx_new (MM_MTX_DENSE, MM_MTX_UNSYMMETRIC, s->m, s->n, s->m * s->n);
	mm_mtx_set_all (d, 0.);

	for (k = 0; k < s->nz; k++) {
		int		i = s->i[k];
		int		j = s->j[k];
		d->data[i + j * s->m] = s->data[k];
	}
	return d;
}

mm_sparse *
mm_mtx_dense_to_sparse (const mm_dense *d, const double threshold)
{
	int			i, j, k;
	mm_sparse	*s;
	if (!mm_is_dense (d->typecode)) {
		cdescent_warning ("mm_dense_to_sparse", "input matrix is not dense.", __FILE__, __LINE__);
		return mm_mtx_copy (d);
	}

	s = mm_mtx_new (MM_MTX_SPARSE, MM_MTX_UNSYMMETRIC, d->m, d->n, d->m * d->n);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->j = (int *) malloc (s->nz * sizeof (int));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));

	k = 0;
	s->p[0] = 0;
	for (j = 0; j < d->n; j++) {
		for (i = 0; i < d->m; i++) {
			double	dij = d->data[i + j * d->m];
			if (fabs (dij) >= threshold) {
				s->i[k] = i;
				s->j[k] = j;
				s->data[k] = dij;
				k++;
			}
		}
		s->p[j + 1] = k;
	}
	return s;
}

static mm_mtx *
mm_mtx_speye (const int n)
{
	int		k;
	mm_mtx	*mm = mm_mtx_new (MM_MTX_SPARSE, MM_MTX_SYMMETRIC, n, n, n);

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
mm_mtx_eye (MM_MtxType type, const int n)
{
	mm_mtx	*mm;

	if (type == MM_MTX_SPARSE) mm = mm_mtx_speye (n);
	else {
		int		k;
		mm = mm_mtx_new (MM_MTX_DENSE, MM_MTX_UNSYMMETRIC, n, n, n * n);
		mm->data = (double *) malloc (mm->nz * sizeof (double));
		mm_mtx_set_all (mm, 0.);
		for (k = 0; k < n; k++) mm->data[k + k * n] = 1.;
	}

	return mm;
}

/* sum_i S(i) */
double
mm_mtx_sum (const mm_mtx *x)
{
	int		k;
	double	sum = 0.;
	for (k = 0; k < x->nz; k++) sum += x->data[k];
	return sum;
}

/* |S| */
double
mm_mtx_asum (const mm_mtx *x)
{
	return dasum_ (&x->nz, x->data, &ione);
}

/* sum_i S(i,j) */
static double
mm_mtx_sj_sum (const int j, const mm_sparse *s)
{
	int		k;
	double	sum = 0.;
	if (j < 0 || s->n <= j) cdescent_error ("mm_mtx_sj_sum", "specified index is invalid", __FILE__, __LINE__);
	for (k = s->p[j]; k < s->p[j + 1]; k++) sum += s->data[k];
	return sum;
}

/* sum_i X(i,j) */
double
mm_mtx_xj_sum (const int j, const mm_mtx *x)
{
	double	sum = 0.;
	if (mm_is_sparse (x->typecode)) sum = mm_mtx_sj_sum (j, x);
	else {
		int		k;
		for (k = 0; k < x->m; k++) sum += x->data[k + j * x->m];
	}
	return sum;
}

/* ||S|| */
double
mm_mtx_nrm2 (const mm_mtx *x)
{
	return dnrm2_ (&x->nz, x->data, &ione);
}

/* ||S(:,j)|| */
static double
mm_mtx_sj_nrm2 (const int j, const mm_sparse *s)
{
	int		size = s->p[j + 1] - s->p[j];
	return dnrm2_ (&size, s->data + s->p[j], &ione);
}

/* ||X(:,j)|| */
double
mm_mtx_xj_nrm2 (const int j, const mm_mtx *x)
{
	if (j < 0 || x->n <= j) cdescent_error ("mm_mtx_nrm2", "index out of range.", __FILE__, __LINE__);
	return (mm_is_sparse (x->typecode)) ? mm_mtx_sj_nrm2 (j, x) : dnrm2_ (&x->m, x->data + j * x->m, &ione);
}

/* S * D or S' * D */
static mm_mtx *
mm_mtx_s_dot_d (bool trans, const double alpha, const mm_sparse *s, const mm_dense *d, const double beta)
{
	int		j, k, l;
	int		m, n, lda;
	bool	symmetric;
	mm_mtx	*mm;

	m = (trans) ? s->n : s->m;
	n = (trans) ? s->m : s->n;
	if (n != d->m) cdescent_error ("mm_mtx_s_dot_d", "matrix size not match.", __FILE__, __LINE__);

	lda = (trans) ? s->n : s->m;

	mm = mm_mtx_new (false, false, m, d->n, m * d->n);

	symmetric = mm_is_symmetric (s->typecode);

	mm->data = (double *) malloc (mm->nz * sizeof (double));
	mm_mtx_set_all (mm, beta);

	for (l = 0; l < d->n; l++) {
		for (j = 0; j < s->n; j++) {
			for (k = s->p[j]; k < s->p[j + 1]; k++) {
				int		si = (trans) ? s->i[k] : s->j[k];
				int		sj = (trans) ? s->j[k] : s->i[k];
				mm->data[sj + l * lda] += alpha * s->data[k] * d->data[si + l * d->m];
				if (symmetric && j != sj) mm->data[si + l * lda] += alpha * s->data[k] * d->data[sj + l * d->m];
			}
		}
	}
	return mm;
}

/* X * y or X' * y */
mm_mtx *
mm_mtx_x_dot_y (bool trans, const double alpha, const mm_mtx *x, const mm_dense *y, const double beta)
{
	int		m = (trans) ? x->n : x->m;
	int		n = (trans) ? x->m : x->n;
	mm_mtx	*c;

	if (n != y->m) cdescent_error ("mm_mtx_x_dot_y", "matrix size not match.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) cdescent_error ("mm_mtx_x_dot_y", "matrix *y must be dense.", __FILE__, __LINE__);

	c = mm_mtx_new (MM_MTX_DENSE, MM_MTX_UNSYMMETRIC, m, y->n, m * y->n);
	if (mm_is_sparse (x->typecode)) c = mm_mtx_s_dot_d (trans, alpha, x, y, beta);
	else {
		c->data = (double *) malloc (m * y->n * sizeof (double));
		dgemv_ ((trans) ? "T" : "N", &x->m, &x->n, &alpha, x->data, &x->m, y->data, &ione, &beta, c->data, &ione);
	}
	return c;
}

/* S(:,j)' * D */
static double
mm_mtx_sj_trans_dot_d (const int j, const mm_sparse *s, const mm_dense *d)
{
	int		k;
	double	val;

	val = 0;
	for (k = s->p[j]; k < s->p[j + 1]; k++) val += s->data[k] * d->data[s->i[k]];
	return val;
}

/* X(:,j)' * y */
double
mm_mtx_xj_trans_dot_y (const int j, const mm_mtx *x, const mm_dense *y)
{
	double	val;

	if (j < 0 || x->n <= j) cdescent_error ("mm_mtx_xj_trans_dot_y", "index out of range.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) cdescent_error ("mm_mtx_xj_trans_dot_y", "matrix *y must be dense.", __FILE__, __LINE__);

	if (mm_is_sparse (x->typecode)) val = mm_mtx_sj_trans_dot_d (j, x, y);
	else val = ddot_ (&x->m, x->data + j * x->m, &ione, y->data, &ione);

	return val;
}

/* d += a * S(:,j) */
static void
mm_mtx_asjpd (const double alpha, const int j, const mm_sparse *s, mm_dense *d)
{
	int		k;
	for (k = s->p[j]; k < s->p[j + 1]; k++) d->data[s->i[k]] += alpha * s->data[k];
	return;
}

/* y += a * X(:,j) */
void
mm_mtx_axjpy (const double alpha, const int j, const mm_mtx *x, mm_dense *y)
{
	if (!mm_is_dense (y->typecode)) cdescent_error ("mm_mtx_axjpy", "matrix *y must be dense.", __FILE__, __LINE__);

	if (mm_is_sparse (x->typecode)) mm_mtx_asjpd (alpha, j, x, y);
	else daxpy_ (&y->m, &alpha, x->data + j * x->m, &ione, y->data, &ione);

	return;
}
