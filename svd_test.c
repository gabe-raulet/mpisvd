#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "kiss.h"
#include "cblas.h"
#include "lapacke.h"
#include "mmio_dense.h"
#include "linalg_routines.h"
#include "svd_routines.h"

int iseed[4];
int iseed_init();
int log2i(int v);

int main(int argc, char *argv[])
{
    kiss_init();
    iseed_init();



    return 0;
}
