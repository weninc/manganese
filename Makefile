.SUFFIXES:

########## change if necessary ###############
CXX = g++

# Debug build
#CXXFLAGS = -Wall -g -std=c++11

#Release build
CXXFLAGS = -Wall -O3 -DNDEBUG -march=native -I/usr/local/opt/hdf5@1.8/include

LIBS = -L/usr/local/opt/hdf5@1.8/lib -lhdf5 -lhdf5_hl -lpthread -lfftw3  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

##############################################

OBJECTS = propagator.o output.o

manganese:	manganese.cpp ${OBJECTS}
	$(CXX) $(CXXFLAGS) $(OPTIONS) manganese.cpp $(OBJECTS) $(LIBS)


%.o:	%.cpp %.h
	$(CXX) $(CXXFLAGS) $(OPTIONS) -c $< -o $@

clean:
	rm *.o
