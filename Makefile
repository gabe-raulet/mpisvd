CC=clang
MPICC=mpicc
INCS=-I/opt/homebrew/Cellar/openblas/0.3.24/include -I/opt/homebrew/include -I./inc
LIBS=-L/opt/homebrew/Cellar/lapack/3.11/lib -L/opt/homebrew/Cellar/openblas/0.3.24/lib
LINKS=-llapacke -lopenblas
OBJS=kiss.o mmio.o mmio_dense.o svd_routines.o svd_serial.o svd_dist.o
PROGS=full_svd mpi_svd
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

test: full_svd
	python run_serial_tests.py

full_svd: full_svd.c $(OBJS)
	$(MPICC) $(CFLAGS) $(INCS) $(LIBS) $(LINKS) -o $@ $^

mpi_svd: mpi_svd.c $(OBJS)
	$(MPICC) $(CFLAGS) $(INCS) $(LIBS) $(LINKS) -o $@ $^

clean:
	rm -rf *.dSYM *.o *.out *.mtx *.diag

distclean: clean
	rm -rf $(PROGS)

mmio.o: inc/mmio.h
mmio_dense.o: inc/mmio_dense.h inc/mmio.h
svd_routines.o: inc/svd_routines.h
svd_serial.o: inc/svd_serial.h
svd_dist.o: inc/svd_dist.h
kiss.o: inc/kiss.h
