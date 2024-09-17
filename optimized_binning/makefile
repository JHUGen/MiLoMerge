CPPFLAGS=-I /home/mohit/deb_packages/eigen-3.4.0/ -I ./ -g -std=c++11 -O2

all:
	g++ bin_splitter.cpp  $(CPPFLAGS) -c -g;\
	g++ test_split.cpp bin_splitter.o $(CPPFLAGS) -o split