PWD=$(shell pwd)
LINKER=g++
CPPFLAGS=-g -O3 -fPIC -fopenmp 
CPPFLAGS+=-Wextra -Wall -Wfloat-equal -Wundef -Wpointer-arith -Wcast-align -Wunreachable-code 
INC=-I/usr/include/eigen3 -I$(PWD)/Inc
SRC=$(PWD)/Src
PYDIR=$(PWD)/Py
OBJDIR=$(PWD)/Obj
TESTDIR=$(PWD)/Test
ROOTFLAGS=$(shell root-config --cflags) $(shell root-config --libs) -Lrootlib
PYFLAGS=$(CPPFLAGS) -shared $(shell python3 -m pybind11 --includes) -o bin_splitter$(shell python3-config --extension-suffix)

all: base py

base: $(SRC)/bin_splitter.cpp
	$(LINKER) $(SRC)/bin_splitter.cpp $(CPPFLAGS) $(INC) -c -o $(OBJDIR)/bin_splitter.o

test_root: $(TESTDIR)/test_split.cpp base
	$(LINKER) $(TESTDIR)/test_split.cpp $(OBJDIR)/bin_splitter.o $(CPPFLAGS) $(INC) -o $(PWD)/test_split $(ROOTFLAGS)

test_local: $(TESTDIR)/test_split_local.cpp base
	$(LINKER) $(TESTDIR)/test_split_local.cpp $(OBJDIR)/bin_splitter.o $(CPPFLAGS) $(INC) -o $(PWD)/test_split

py: $(PYDIR)/bin_split_bind.cpp base
	$(LINKER) $(INC) $(CPPFLAGS) $(PYDIR)/bin_split_bind.cpp $(OBJDIR)/bin_splitter.o $(PYFLAGS)

