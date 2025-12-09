/* ***************************************************************************** */
/* Copyright:      Francois Panneton and Pierre L'Ecuyer, University of Montreal */
/*                 Makoto Matsumoto, Hiroshima University                        */
/* Notice:         This code can be used freely for personal, academic,          */
/*                 or non-commercial purposes. For commercial purposes,          */
/*                 please contact P. L'Ecuyer at: lecuyer@iro.UMontreal.ca       */
/*                 A modified "maximally equidistributed" implementation         */
/*                 by Shin Harase, Hiroshima University.                         */
/* ***************************************************************************** */

void InitWELLRNG19937(unsigned int *init)
extern unsigned int (*WELLRNG19937)(void);
