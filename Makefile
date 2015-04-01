# Makefile for neural2d.
#
# neural2d is the standalone console program for the neural net.
#
# This Makefile has the following targets:
#
#    make all        # same as make neural2d
#    make            # defaults to make all
#    make clean      # removes the neural2d object files
#    make test       # execute neural2d on a test set of data (digits demo)
#    make test-xor   # train neural2d to do the XOR function
#    make test-mnist # train on the MNIST handwritten digits image set


# Specify a compiler invocation that understands C++11 with whatever options it requires:
COMPILER=g++ -std=c++11 -pthread

# Non-essential options, e.g., add -fopenmp to enable OpenMP
EXTRACFLAGS=-g -O2 -Wall -Wextra -pedantic -Wno-missing-field-initializers


all: neural2d

# The next rules make the neural2d program.

neural2d: neural2d.o neural2d-core.o webserver.o Makefile
	$(COMPILER) $(EXTRACFLAGS) neural2d.o neural2d-core.o webserver.o -o neural2d

neural2d.o: neural2d.cpp neural2d.h Makefile
	$(COMPILER) $(EXTRACFLAGS) -c neural2d.cpp -o neural2d.o

neural2d-core.o: neural2d-core.cpp neural2d.h Makefile
	$(COMPILER) $(EXTRACFLAGS) -c neural2d-core.cpp -o neural2d-core.o

webserver.o: webserver.cpp neural2d.h webserver.h Makefile
	$(COMPILER) $(EXTRACFLAGS) -c webserver.cpp -o webserver.o

clean:
	rm neural2d neural2d.o neural2d-core.o webserver.o

# Run neural2d on a test set of images (the easy digits demo):

images/digits/test-1.bmp:
	@echo "Before running the test, you must extract the archive of images in images/digits/"
	@false

test: images/digits/test-1.bmp
	./neural2d images/digits/topology.txt images/digits/inputData.txt images/digits/weights.txt

# XOR test:

test-xor: topology-xor.txt inputData-xor.txt
	./neural2d topology-xor.txt inputData-xor.txt weights.txt

# MNIST handwritten digits test:

images/mnist/topology-mnist.txt:
	@echo "Missing images/mnist/topology-mnist.txt"
	@false

images/mnist/inputData-mnist.txt:
	@echo "You must first create images/mnist/inputData-mnist.txt"
	@echo "Attempting to do that automatically..."
	cd images/mnist && ./makeTrainDataForNeural2d.py || python ./makeTrainDataForNeural2d.py

images/mnist/train-data/0.bmp:
	@echo "Before running the MNIST test, you must extract the BMP images into images/mnist/train-data/"
	@echo "Attempting to do that automatically..."
	cd images/mnist && ./makeTrainDataForNeural2d.py || python ./makeTrainDataForNeural2d.py

test-mnist: images/mnist/topology-mnist.txt images/mnist/inputData-mnist.txt images/mnist/train-data/0.bmp
	./neural2d images/mnist/topology-mnist.txt images/mnist/inputData-mnist.txt weights-mnist.txt


.PHONY: all clean test test-xor test-mnist
