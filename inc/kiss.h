#ifndef KISS_H_
#define KISS_H_

/*
 * Reference: http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
 */

#include <stdlib.h>

/*
 * @func dev_rand - Generates a random 32-bit unsigned integer using the
 * /dev/urandom system utility. This is useful for seeding other random
 * number generators.
 */
uint32_t dev_rand(void);

/*
 * @func kiss_rand - Generates a pseudo-random 32-bit unsigned integer using
 * the KISS (Keep It Simple Stupid) algorithm. The period of this PRNG is
 * around 2^127.
 */
uint32_t kiss_rand(void);

/*
 * @func kiss_init - Initializes KISS generator using /dev/urandom (dev_rand()).
 */
void kiss_init(void);

/*
 * @func kiss_seed - Initializes KISS generator using seed.
 */
void kiss_seed(uint32_t seed);

/*
 * @func kiss_unirandf - Generates uniformly random double-precision (64 bits) floating
 * point number in the interval between 0 and 1.
 */
double kiss_unirandf(void);


/*
 * @func kiss_gaussrand - Generates a random gaussian (normal) variable with
 * mean 0 and standard deviation 1.
 */
double kiss_gaussrand(void);


#endif
