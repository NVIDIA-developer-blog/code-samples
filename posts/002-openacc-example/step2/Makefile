CC       = pgcc
CCFLAGS  = -I../common
ACCFLAGS = -acc -ta=nvidia -Minfo=accel -lpgacc
OMPFLAGS = -fast -mp -Minfo

BIN =  laplace2d_omp laplace2d_acc

all: $(BIN)

laplace2d_acc: laplace2d.c
	$(CC) $(CCFLAGS) $(ACCFLAGS) -o $@ $<

laplace2d_omp: laplace2d.c
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ $<

clean:
	$(RM) $(BIN)
