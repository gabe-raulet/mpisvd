#include "kiss.h"
#include <unistd.h>
#include <fcntl.h>
#include <math.h>

static uint32_t x = 123456789;
static uint32_t y = 987654321;
static uint32_t z = 43219876;
static uint32_t c = 6543217;

uint32_t dev_rand(void)
{
    int fn;
    uint32_t r;

    fn = open("/dev/urandom", O_RDONLY);

    if (fn == -1) exit(-1);

    if (read(fn, &r, 4) != 4) exit(-1);

    close(fn);

    return r;

}

uint32_t kiss_rand(void)
{
    uint64_t t;

    x = 314527869 * x + 1234567;

    y ^= y << 5;
    y ^= y >> 7;
    y ^= y << 22;

    t = 4294584393ULL * z + c;
    c = t >> 32;
    z = t;

    return x + y + z;
}

void kiss_seed(uint32_t seed)
{
    seed++;

    x = seed;
    y = 123*seed + 234;
    z = 6432*seed;

    c = (seed + 86543331) % 698769068 + 1;
}

void kiss_init(void)
{
    x = dev_rand();
    while (!(y = dev_rand())); /* y cannot be zero */
    z = dev_rand();

    c = dev_rand() % 698769068 + 1;
}

double kiss_unirandf(void)
{
    double v;
    uint32_t a, b;

    a = kiss_rand() >> 6; /* upper 26 bits */
    b = kiss_rand() >> 5; /* upper 27 bits */

    v = (a * 134217728.0 + b) / 9007199254740992.0;

    return v;
}

double kiss_gaussrand(void)
{
    double x2, y2, r;

    do
    {
        x2 = 2. * kiss_unirandf() - 1.;
        y2 = 2. * kiss_unirandf() - 1.;
        r = x2 * x2 + y2 * y2;
    } while (r == 0. || r >= 1.);

    r = sqrt((-2. * log(r)) / r);

    return x2 * r;
}
