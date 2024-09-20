CPPFLAGS=-I/usr/include/eigen3 -I./ -g -O2 $(shell root-config --cflags) $(shell root-config --libs) -Lrootlib

all:
	g++ bin_splitter.cpp  $(CPPFLAGS) -c -g;\
	g++ test_split.cpp bin_splitter.o $(CPPFLAGS) -o split;\
	g++ bin_split_bind.cpp bin_splitter.cpp $(CPPFLAGS) -Wall -shared \
	$(shell python3 -m pybind11 --includes) \
	-o bin_splitter$(shell python3-config --extension-suffix) \
	-fno-var-tracking-assignments -fPIC
