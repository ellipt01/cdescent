/*
 * bic.h
 *
 *  Created on: 2015/03/13
 *      Author: utsugi
 */

#ifndef BIC_H_
#define BIC_H_

#ifdef __cplusplus
extern "C" {
#endif

/* bic.c */
bic_info	*cdescent_eval_bic (const cdescent *cd);

#ifdef __cplusplus
}
#endif

#endif /* BIC_H_ */
