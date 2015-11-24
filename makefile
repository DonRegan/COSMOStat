CC=mpic++
CFLAGS=-O3 -c -Wall -fopenmp
LDFLAGS=-O3 -L$$FFTWDIR -lfftw3 -fopenmp
SOURCES=zbox2.cpp omp_cosmoStat.cpp util.cpp load_snapshot.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=zbox2

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o