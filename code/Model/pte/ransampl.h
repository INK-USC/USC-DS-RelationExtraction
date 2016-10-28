/*
 * Library:   ransampl (random number sampling)
 *
 * File:      ransampl.h
 *
 * Contents:  Random-number sampling using the Walker-Vose alias method
 *
 * Copyright: Joachim Wuttke, Forschungszentrum Juelich GmbH (2013)
 *
 * License:   see ../COPYING (FreeBSD)
 * 
 * Homepage:  apps.jcns.fz-juelich.de/ransampl
 */

#ifndef RANSAMPL_H
#define RANSAMPL_H
#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS /* empty */
# define __END_DECLS /* empty */
#endif
__BEGIN_DECLS

typedef long long integer;

typedef struct {
    integer n;
    integer* alias;
    double* prob;
} ransampl_ws;

ransampl_ws* ransampl_alloc( integer n );

void ransampl_set( ransampl_ws *ws, double *p );

integer ransampl_draw( ransampl_ws *ws, double ran1, double ran2 );

void ransampl_free( ransampl_ws *ws );

__END_DECLS
#endif /* RANSAMPL_H */
