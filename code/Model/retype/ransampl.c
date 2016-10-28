/*
 * Library:   ransampl (random number sampling)
 *
 * File:      ransampl.c
 *
 * Contents:  Random-number sampling using the Walker-Vose alias method,
 *            as described by Keith Schwarz (2011)
 *            [http://www.keithschwarz.com/darts-dice-coins]
 *
 * Copyright: Joachim Wuttke, Forschungszentrum Juelich GmbH (2013)
 *
 * License:   see ../COPYING (FreeBSD)
 * 
 * Homepage:  apps.jcns.fz-juelich.de/ransampl
 */

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "ransampl.h"

//! Allocate workspace for random-number sampling.
ransampl_ws* ransampl_alloc( integer n )
{
    ransampl_ws *ws;
	ws = (ransampl_ws *)malloc(sizeof(ransampl_ws));
	ws->alias = (integer *)malloc(n*sizeof(integer));
	ws->prob = (double *)malloc(n*sizeof(double));
	if (ws == NULL || ws->alias == NULL || ws->prob == NULL)
	{
        fprintf( stderr, "ransampl: workspace allocation failed\n" );
        exit(ENOMEM);
    }
    ws->n = n;
    return ws;
}

//! Initialize workspace by precompute alias tables from given probabilities.
void ransampl_set( ransampl_ws *ws, double* p )
{
    integer n = ws->n;
    integer i, a, g;

    // Local workspace:
    double *P;
    integer *S, *L;
    if ( !(P = (double*) malloc( n*sizeof(double) ) ) ||
         !(S = (integer*) malloc( n*sizeof(integer) ) ) ||
         !(L = (integer*) malloc( n*sizeof(integer) ) ) ) {
        fprintf( stderr, "ransampl: temporary allocation failed\n" );
        exit(ENOMEM);
    }

    // Normalise given probabilities:
    double sum=0;
    for ( i=0; i<n; ++i ) {
        if( p[i]<0 ) {
            fprintf( stderr, "ransampl: invalid probability p[%d]<0\n", (int)(i) );
            exit(EINVAL);
        }
        sum += p[i];
    }
    if ( !sum ) {
        fprintf( stderr, "ransampl: no nonzero probability\n" );
        exit(EINVAL);
    }
    for ( i=0; i<n; ++i )
        P[i] = p[i] * n / sum;

    // Set separate index lists for small and large probabilities:
    integer nS = 0, nL = 0;
    for ( i=n-1; i>=0; --i ) {
        // at variance from Schwarz, we revert the index order
        if ( P[i]<1 )
            S[nS++] = i;
        else
            L[nL++] = i;
    }

    // Work through index lists
    while ( nS && nL ) {
        a = S[--nS]; // Schwarz's l
        g = L[--nL]; // Schwarz's g
        ws->prob[a] = P[a];
        ws->alias[a] = g;
        P[g] = P[g] + P[a] - 1;
        if ( P[g] < 1 )
            S[nS++] = g;
        else
            L[nL++] = g;
    }

    while ( nL )
        ws->prob[ L[--nL] ] = 1;

    while ( nS )
        // can only happen through numeric instability
        ws->prob[ S[--nS] ] = 1;

    // Cleanup:
    free( P );
    free( S );
    free( L );
}

//! Draw one random index, using two supplied uniform random numbers.
integer ransampl_draw( ransampl_ws *ws, double ran1, double ran2 )
{
    integer i = (integer) ws->n * ran1;
    return ran2 < ws->prob[i] ? i : ws->alias[i];
}

//! Free the random-number sampling workspace.
void ransampl_free( ransampl_ws *ws )
{
    free( ws->alias );
    free( ws->prob );
    free( ws );
}
