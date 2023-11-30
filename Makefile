CC=clang
MPICC=mpicc
INCS=-I/opt/homebrew/Cellar/openblas/0.3.24/include -I/opt/homebrew/include -I./inc
LIBS=-L/opt/homebrew/Cellar/lapack/3.12.0/lib -L/opt/homebrew/Cellar/openblas/0.3.24/lib
LINKS=-llapacke -lopenblas
OBJS=kiss.o mmio.o mmio_dense.o svd_routines.o
PROGS=serial_svd
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

serial_svd: serial_svd.c $(OBJS)
	$(MPICC) $(CFLAGS) $(INCS) $(LIBS) $(LINKS) -o $@ $^

clean:
	rm -rf *.dSYM *.o *.out *.mtx *.diag

distclean: clean
	rm -rf $(PROGS)

mmio.o: inc/mmio.h
mmio_dense.o: inc/mmio_dense.h inc/mmio.h
svd_routines.o: inc/svd_routines.h
kiss.o: inc/kiss.h
