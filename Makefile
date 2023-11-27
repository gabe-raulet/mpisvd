CC=clang
MPICC=mpicc
INCS=-I/opt/homebrew/Cellar/openblas/0.3.24/include -I/opt/homebrew/include -I./inc
LIBS=-L/opt/homebrew/Cellar/lapack/3.11/lib -L/opt/homebrew/Cellar/openblas/0.3.24/lib
LINKS=-llapacke -lopenblas
OBJS=kiss.o mmio.o mmio_dense.o linalg_routines.o svd_routines.o
PROGS=rand_svd
CFLAGS=-Wall

D?=0

ifeq ($(D), 1)
CFLAGS+=-O0 -g -fsanitize=address -fno-omit-frame-pointer
else
CFLAGS+=-O2
endif

%.o: src/%.c
	$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

all: $(PROGS)
	rm -rf *.o

rand_svd: rand_svd.c $(OBJS)
	$(CC) $(CFLAGS) $(INCS) $(LIBS) $(LINKS) -o $@ $^

clean:
	rm -rf *.dSYM *.o *.out

distclean: clean
	rm -rf $(PROGS) *.mtx

mmio.o: inc/mmio.h
mmio_dense.o: inc/mmio_dense.h inc/mmio.h
linalg_routines.o: inc/linalg_routines.h
svd_routines.o: inc/svd_routines.h inc/linalg_routines.h
kiss.o: inc/kiss.h
