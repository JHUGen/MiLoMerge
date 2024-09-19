CPPFLAGS=-I/usr/include/eigen3 -I./ -g -O2 $(shell root-config --cflags) $(shell root-config --libs) -Lrootlib

all:
	g++ bin_splitter.cpp  $(CPPFLAGS) -c -g;\
	g++ test_split.cpp bin_splitter.o $(CPPFLAGS) -o split
