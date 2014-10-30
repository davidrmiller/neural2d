# Makefile for neural2d.
#
# neural2d is the standalone console program for the neural net.
# neural2d-gui is the Python GUI for neural2d.
#
# neural2d.cpp + neural2d-core.cpp + neural2d.h ==> neural2d

MYCFLAGS=-g -O1
CFLAGS=$(MYCFLAGS) -std=c++11 -Wall -fopenmp

all: neural2d qtui.py

# This rule makes the QT4 part of neural-net-control.py. If the controller
# program is not needed, this rule can be disabled.

qtui.py: qtui-neural-net2.ui
	pyuic4 qtui-neural-net2.ui > qtui.py

# The next rules make the neural-net program. It can be run standalone without
# a controlling program, or spawned and managed by the controlling program.

# Warning: -O2 and -fopenmp do not work well together.
# It's ok to use -O1 and -fopenmp at the same time.

neural2d: neural2d.o neural2d-core.o Makefile
	g++ $(CFLAGS) neural2d.o neural2d-core.o -o neural2d

neural2d.o: neural2d.cpp neural2d.h Makefile
	g++ $(CFLAGS) -c neural2d.cpp -o neural2d.o

neural2d-core.o: neural2d-core.cpp neural2d.h Makefile
	g++ $(CFLAGS) -c neural2d-core.cpp -o neural2d-core.o

clean:
	rm neural2d neural2d.o neural2d-core.o

