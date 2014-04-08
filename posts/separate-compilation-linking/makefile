objects = main.o particle.o v3.o

all: $(objects)
		nvcc -arch=sm_20 $(objects) -o app

%.o: %.cpp
		nvcc -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
		rm -f *.o app