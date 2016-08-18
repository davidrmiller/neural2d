Neural2d - Neural Net Simulator
================================

User Manual
===========

Ver. 1.0  
=======

Updated 18-Aug-2016

Intro video (11 min): 
[https://www.youtube.com/watch?v=yB43jj-wv8Q](https://www.youtube.com/watch?v=yB43jj-wv8Q)

Blog: [neural2d.net](http://neural2d.net)

Features
--------

* Optimized for 2D input data
* Neuron layers can be abstracted as 1D or 2D arrangements of neurons
* Input data can binary or text
* Network topology is defined in a text file
* Neurons in layers can be fully or sparsely connected
* Selectable transfer function per layer
* Adjustable or automatic training rate (eta)
* Optional momentum (alpha) and regularization (lambda)
* Convolution filtering and convolution networking
* Standalone console program
* Simple, heavily-commented code, suitable for prototyping, learning, and experimentation
* Optional web-browser-based GUI controller
* Graphic visualizations of hidden-layer data
* No dependencies! Just C++11 (and POSIX networking for the optional webserver interface)

Document Contents
-----------------

[Overview](#Overview)  
[Requirements](#Requirements)  
[Compiling the source](#Compiling)  
[How to run the digits demo](#Demo)  
[How to run the XOR example](#XorExample)  
[GUI interface](#GUI)  
[How to use your own data](#YourOwnData)  
[The 2D in neural2d](#2D)  
[Convolution filtering](#ConvolutionFiltering)  
[Convolution networking and pooling](#ConvolutionNetworking)  
[Layer depth](#LayerDepth)  
[Topology config file format](#topologyconfig)  
[Topology config file examples](#TopologyExamples)  
[How-do-I *X*?](#HowDoI)  
* [How do I run the command-line program?](#howConsole)  
* [How do I run the GUI interface?](#howGui)  
* [How do I disable the GUI interface?](#howDisableGui)  
* [How do I use my own data instead of the digits images?](#howOwnData)  
* [How do I use a trained net on new data?](#howTrained)  
* [How do I train on the MNIST handwritten digits data set?](#MNIST)  
* [How do I change the learning rate parameter?](#howEta)  
* [Are the output neurons binary or floating point?](#howBinary)  
* [How do I use a different transfer function?](#howTf)  
* [How do I define a convolution filter?](#howConvolve)  
* [How do I define convolution networking and pooling?](#howConvolveNetworking)  
* [How do the color image pixels get converted to floating point for the input layer?](#howRgb)  
* [How can I use .jpg and .png images as inputs to the net?](#howJpg)  
* [How do I find my way through the source code?](#sourcecode)  
* [Why does the net error rate stay high? Why doesn't my net learn?](#howLearn)  
* [What other parameters do I need to know about?](#howParams)  



Overview<a name="Overview"></a>
------------

Neural2d is a standalone console program with optional HTTP web
interface written in C++. It's a backpropagation neural net simulator,
with features that make it easy to think of your input data as either
one-dimensional or two-dimensional. You specify a network topology in
a text file (topology.txt). The input data to the neural is specified
in a text file (inputData.txt). The inputData.txt file can contain the
actual input values for all the input samples, or it can contain a list
of .bmp or .dat files that contain the input data in binary form.

Neural2d is for educational purposes. It's not production-quality code,
and it's not burdened with a lot of bling. It's just heavily-commented
neural net code that you can take apart and see how it works. Or modify
it and experiment with new ideas. Or extract the functions you need and
[embed them](#Licenses) into your own project.

If you're not using the optional GUI interface, then neural2d has no
dependencies other than a conforming C++11 compiler. If using the GUI
interface, you'll need to link with a standard POSIX sockets networking
library.




Requirements<a name="Requirements"></a>
------------

* C++-11 compiler
  * e.g., g++ 4.7 on Linux or Cygwin, Visual Studio 2013 on Windows
* POSIX sockets (only needed if compiling the optional GUI)
  * e.g., Cygwin on Windows
* CMake 2.8.12 or later
* Compiles and runs on Linux, Windows, and probably Mac




Compiling the source<a name="Compiling"></a>
------------

We use CMake to configure the build system. First get the source code
from the Gitub repository. If using the command line, the command is:

     git clone https://github.com/davidrmiller/neural2d

That will put the source code tree into a directory named neural2d.

### Compiling with CMake graphical interface

If you are using the CMake graphical interface, run it and set the
"source" directory to the neural2d top directory, and set the binary
output directory to a build directory under that (you must create the
build directory), then click Configure and Generate. Uncheck WEBSERVER
if you don't want to compile the optional GUI. Here is what it looks like:

![CMake GUI example](https://raw.github.com/davidrmiller/neural2d/master/images/cmake-gui.png)

### Compiling with CMake command line interface

If you are using CMake from the command line, cd to the neural2d top
level directory, make a build directory, then run cmake from there:

```
git clone https://github.com/davidrmiller/neural2d
cd neural2d
mkdir build
cd build
cmake ..
make
```

There is no "install" step. After the neural2d program is compiled,
you can execute it or open the project file from the build directory.

On Windows, by default CMake generates a Microsoft Visual Studio project
file in the build directory. On Linux and Cygwin, CMake generates
a Makefile that you can use to compile neural2d. You can specify a
different CMake generator with the -G option, for example:

     cmake -G "Visual Studio 14 2015" ..

To get a list of available CMake generators:

     cmake --help

If you get errors when compiling the integrated webserver, you can
build neural2d without webserver support by running CMake with the
-DWEBSERVER=OFF option, like this:

     cmake -DWEBSERVER=OFF ..





How to run the digits demo<a name="Demo"></a>
-----------

On systems using Makefiles, in the build directory, execute:

    make test

This will do several things: it will compile the
neural2d program if necessary; it will expand the
archive named image/digits/digits.zip into [5000 individual
images](https://github.com/davidrmiller/neural2d/wiki/DemoDigits);
and it will then train the neural net to classify those digit images.

The input data, or "training set," consists of images of numeric
digits. The first 50 look like these:

![training samples](https://raw.github.com/davidrmiller/neural2d/master/images/digits-illus.png)

The images are 32x32 pixels each, stored in .bmp format. In this demo,
the neural net is configured to have 32x32 input neurons, and 10 output
neurons. The net is trained to classify the digits in the images and
to indicate the answer by driving the corresponding output neuron to a
high level.

Once the net is sufficiently trained, all the connection weights are
saved in a file named "weights.txt".

If you are not using Makefiles, you will need to expand the archive in
images/digits, then invoke the neural2d program like this:

     neural2d ../images/digits/topology.txt ../images/digits/inputData.txt weights.txt





How to run the XOR example<a name="XorExample"></a>
--------------------------

On systems using Makefiles, in the build directory, execute:

     make test-xor

For more information about the XOR example, see
[this wiki page](https://github.com/davidrmiller/neural2d/wiki/XOR_Example).





GUI interface (optional)<a name="GUI"></a>
-------------

First, launch the neural2d console program in a command window with the -p option:

     ./neural2d topology.txt inputData.txt weights.txt -p

The -p option causes the neural2d program to wait for a command before
starting the training. The screen will look something like this:

![console-window](https://raw.github.com/davidrmiller/neural2d/master/images/console1.png)

At this point, the neural2d console program is paused and waiting for
a command to continue. Using any web browser, open:

     http://localhost:24080

A GUI interface will appear that looks like:

![HTTP interface](https://raw.github.com/davidrmiller/neural2d/master/images/gui2-sm.png)

Press Resume to start the neural net training. It will automatically
pause when the average error rate falls below a certain threshold (or
when you press Pause). You now have a trained net. You can press Save
Weights to save the weights for later use.

See the neural2d wiki for 
[design notes on the web interface](https://github.com/davidrmiller/neural2d/wiki/WebServer).


### Visualizations

At the bottom of the GUI window, a drop-down box shows the visualization
options that are available for your network topology, as shown
below. There will be options to display the activation (the outputs)
of any 2D layer of neurons 3x3 or larger, and convolution kernels of
size 3x3 or larger. Visualization images appear at the bottom of the
GUI. You can mouse-over the images to zoom in.

![visualizerExample](https://raw.github.com/davidrmiller/neural2d/master/images/visualizerExample.png)





How to use your own data<a name="YourOwnData"></a>
------------------------

Input data to the neural net can be specified in text or binary format. If
you want to specify input values in text format, just list the input
values in the inputData.txt file. If you're inputting from .bmp or .dat
binary files, then list those filenames in inputData.txt. Details are
explained below.


### Text input format

To specify input data as text, prepare a text file, typically called
inputData.txt, with one input sample per line. The input values go inside
curly braces. If the data is for training, the target output values must
be specified on the same line after the input values. In each sample, the
input values are given in a linear list, regardless whether the neural
net input layer is one-dimensional or two-dimensional. For example, if
your neural net has 9 input neurons of any arrangement and two output
neurons, then the inputData.txt file for training will look like this
format:

    { 0.32 0.98 0.12 0.44 0.98 0.22 0.34 0.72 0.84 } -1  1   
    { 1.00 0.43 0.19 0.83 0.97 0.87 0.75 0.47 0.92 }  1  1   
    { 0.87 0.75 0.47 0.92 1.00 0.43 0.19 0.83 0.97 } -1 -1   
    { 0.34 0.83 0.97 0.87 0.75 0.43 0.19 0.47 0.92 } -1  1   
    etc.. . .  


### Binary input formats

There are two binary options -- .bmp image files, and .dat data
files. First, prepare your .bmp or .dat files, one sample per file. If
using .bmp files, the number of pixels in each image should equal the
number of input neurons in your neural net. If using .dat files, each
file must contain a linear list of input values, with the same number
of values as the number of input neurons in your neural net.

Next, prepare an input data configuration file, called inputData.txt
containing a list of the .bmp or .dat filenames, one per line. If the
data is for training, then also list the target output values for each
input sample after the filename, like this:

    images/thumbnails/test-918.bmp -1 1 -1 -1 -1 -1 -1 -1 -1 -1
    images/thumbnails/test-919.bmp -1 -1 -1 -1 -1 -1 -1 -1 1 -1
    images/thumbnails/test-920.bmp -1 -1 -1 -1 -1 -1 1 -1 -1 -1
    images/thumbnails/test-921.bmp -1 -1 -1 -1 -1 1 -1 -1 -1 -1
    etc. . .

The path and filename cannot contain any spaces.

The path_prefix directive can be used to specify a string to be attached
to the front of all subsequent filenames, or until the next path_prefix
directive. For example, the previous example could be written:

     path_prefix = images/thumbnails/
     test-918.bmp
     test-919.bmp
     test-920.bmp
     test-921.bmp
     etc. . .

For more information on the .bmp file format, see [this Wikipedia
article.](https://en.wikipedia.org/wiki/BMP_file_format).

For more information on the neural2d .dat binary format, see [this wiki
page.](https://github.com/davidrmiller/neural2d/wiki/.dat-input-file-format)
and the [design
notes.](https://github.com/davidrmiller/neural2d/issues/21)



Topology file<a name="TopologyFile"></a>
---------------------

In addition to the input data config file (inputData.txt), you'll also
need a topology config file (typically named topology.txt by default)
to define your neural net topology (the number and arrangement
of neurons and connections). Its format is described in a [later
section](#topologyconfig).  A typical one looks like this example:

    input size 32x32  
    layer1 size 32x32 from input radius 8x8  
    layer2 size 16x16 from layer1  
    output size 1x10 from layer2  

Then run neuron2d (optionally with the web browser interface) and
experiment with the parameters until the net is adequately trained,
then save the weights in a file for later use.

If you run the web interface, you can change the global parameters
from the GUI while the neural2d program is running. If you run the
neural2d console program without the GUI interface, there is no way
to interact with it while running. Instead, you'll need to examine and
modify the parameters in the code at the top of the files neural2d.cpp
and neural2d-core.cpp.





The 2D in neural2d<a name="2D"></a>
------------------

In a simple traditional neural net model, the neurons are arranged in
a column in each layer:

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-542-sm.png)

In neural2d, you can specify a rectangular arrangement of neurons in
each layer, such as:

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-5x5-4x4-2-sm.png)

The neurons can be sparsely connected to mimic how retinal neurons
are connected in biological brains. For example, if a radius of "1x1"
is specified in the topology config file, each neuron on the right
(destination) layer will connect to a circular patch of neurons in the
left (source) layer as shown here (only a single neuron on the right
side is shown connected in this picture so you can see what's going on,
but imagine all of them connected in the same pattern):

![radius-1x1](https://raw.github.com/davidrmiller/neural2d/master/images/proj-1x1-sm.png )

The pattern that is projected onto the source layer is elliptical. (Layers
configured as convolution filters work slightly differently; see the
later section about convolution filtering.)

Here are some projected connection patterns for various radii:

radius 0x0   
![](https://raw.github.com/davidrmiller/neural2d/master/images/radius-0x0.png)

radius 1x1   
![](https://raw.github.com/davidrmiller/neural2d/master/images/radius-1x1.png)

radius 2x2   
![](https://raw.github.com/davidrmiller/neural2d/master/images/radius-2x2.png)

radius 3x1   
![](https://raw.github.com/davidrmiller/neural2d/master/images/radius-3x1.png)





Convolution filtering<a name="ConvolutionFiltering"></a>
---------------------

Any layer other than the input layer can be configured as a convolution
filter layer by specifying a *convolve-matrix* specification for the
layer in the topology config file.  The neurons are still called neurons,
but their operation differs in the following ways:

* The connection pattern to the source layer is defined by the convolution
matrix (kernel) dimensions (not by a *radius* parameter)

* The connection weights are initialized from the convolution matrix,
and are constant throughout the life of the net.

* The transfer function is automatically set to the identity function
(not by a *tf* parameter).

For example, the following line in the topology config file defines a
3x3 convolution matrix for shrarpening the source layer:

     layerConv1 size 64x64 from input convolve {{0,-1,0},{-1,5,-1},{0,-1,0}}

When a convolution matrix is specified for a layer, you cannot also
specify a *radius* parameter for that layer, as the convolution matrix
size determines the size and shape of the rectangle of neurons in the
source layer. You also cannot also specify a *tf* parameter, because the
transfer function on a convolution layer is automatically set to be the
identity function.

The elements of the convolution matrix are stored as connection weights
to the source neurons. Connection weights on convolution layers are not
updated by the back propagation algorithm, so they remain constant for
the life of the net.

For illustrations of various convolution kernels, see
[this Wikipedia article](http://en.wikipedia.org/wiki/Kernel_%28image_processing%29)

In the following illustration, the topology config file defines a
convolution filter with a 2x2 kernel that is applied to the input layer,
then the results are combined with a reduced-resolution fully-connected
pathway. The blue connections in the picture are the convolution
connections; the green connections are regular neural connections:

    input size 8x8
    layerConvolve size 8x8 from input convolve {{-1,2},{-1,2}}
    layerReducedRes size 4x4 from input
    output size 2 from layerConvolve
    output size 2 from layerReducedRes

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-convolve-8x8.png)





Convolution networking and pooling<a name="ConvolutionNetworking"></a>
---------------------

A **[convolution network
layer](http://en.wikipedia.org/wiki/Convolutional_neural_network)** is
like a convolution filter layer, except that the kernel participates in
backprop training, and everything inside the layer is replicated *N* times
to train *N* separate kernels. A convolution network layer is said to have
depth *N*. A convolution network layer has *depth* \* *X* \* *Y* neurons.

Any layer other than the input or output layer can be configured as
a convolution networking layer by specifying a layer depth > 1, and
specifying the kernel size with a convolve parameter. For example,
to train 40 kernels of size 7x7 on an input image of 64x64 pixels:

      input size 64x64
      layerConv size 40*64x64 from input convolve 7x7
      . . .

A **[pooling layer](http://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer)** 
down-samples the previous layer by finding the average or maximum in
patches of source neurons.  A pooling layer is defined in the topology
config file by specifying a pool parameter on a layer.

In the topology config syntax, the pool parameter requires the argument
"avg" or "max" followed by the operator size, For example, in a
convolution network pipeline of depth 20, you might have these layers:

      input size 64x64
      layerConv size 20*64x64 from input convolve 5x5
      layerPool size 20*16x16 from layerConv pool max 4x4
      . . .





Layer depth<a name="LayerDepth"></a>
---------------------

All layers have a depth, whether explicit or implicit. Layer depth is
specified in the topology config file in the layer size parameter by an
integer and asterisk before the layer size. If the depth is not specified,
it defaults to one. For example:

* size 10*64x64 means 64x64 neurons, depth 10

* size 64x64 means 64x64 neurons, depth 1

* size 1*64x64 also means 64x64 neurons, depth 1

* size 10*64 means 64x1 neurons, depth 10 (the Y dimension defaults to 1)

The primary purpose of layer depth is to allow convolution network
layers to train multiple kernels.  However, the concept of layer depth
is generalized in neural2d, allowing any layer to have any depth and
connect to any other layer of any kind with any depth.

The way two layers are connected depends on the relationship of the
source and destination layer depths as shown below:

 Relationship | How connected
------------- | -----------------
src depth == dst depth  | connect only to the corresponding depth in source
src depth != dst depth  | fully connect across all depths





Topology config file format<a name="topologyconfig"></a>
---------------------------

Here is the grammar of the topology config file:

> *layer-name* *parameters* := *parameter* [ *parameters* ]

> *parameters* := *parameter* [ *parameters* ]

> *parameter* :=

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input | output | layer*name*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;size *dxy-spec*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;from *layer-name*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;channel *channel-spec*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;radius *xy-spec*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tf *transfer-function-spec*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;convolve *filter-spec*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;convolve *xy-spec*

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pool { max | avg } *xy-spec*

> *dxy-spec* := [ *integer* \* ] *integer* [ x *integer* ]

> *xy-spec* := *integer* [ x *integer* ]
 
> *channel-spec* := R | G | B | BW

> *transfer-function-spec* := tanh | logistic | linear | ramp | gaussian | relu
 
> *filter-spec* := same {{,},{,}} syntax used for array initialization in C, C#, VB, Java, etc.

Rules:

1. Comment lines that begin with "#" and blank lines are ignored.

1. The first layer defined must be named "input".

1. The last layer defined must be named "output".

1. The hidden layers can be named anything beginning with "layer".

1. The argument for "from" must be a layer already defined.

1. The color channel parameter can be specified only on the input layer.

1. If a size parameter is omitted, the size is copied from the layer
specified in the from parameter.

1. A radius parameter cannot be used on the same line with a convolve
or pool parameter.

1. The same layer name can be defined multiple times with different
"from" parameters.  This allows source neurons from more than one layer
to be combined in one destination layer. The source layers can be any
size, but the repeated (the destination) layer must have the same size
in each specification. For example, in the following, layerCombined is
size 16x16 and takes inputs from two source layers of different sizes:

```
     input size 128x128  
     layerVertical size 32x32 from input radius 1x8  
     layerHorizontal size 16x16 from input radius 8x1  
     layerCombined from layerHorizontal          <= assumes size 16x16 from the source layer  
     layerCombined size 16x16 from layerVertical <= repeated destination, must match 16x16  
     output size 1 from layerCombined  
```

1. In the *xy-spec*  and in the X,Y part of the *dxy-spec*, you may
specify one or two dimensions.  Spaces are not allowed in the size
spec. If only one dimension is given, the other is assumed to be 1.
For example:

 * "8x8" means 64 neurons in an 8 x 8 arrangement.  
 * "8x1" means a row of 8 neurons
 * "1x8" means a column of 8 neurons.  
 * "8" means the same as "8x1"  





Topology config file examples<a name="TopologyExamples"></a>
-----------------------------

Here are a few complete topology config files and the nets they specify.

    input size 4x4
    layer1 size 3x3 from input
    layer2 size 2x2 from layer1
    output size 1 from layer2

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-4x4-3x3-2x2-1-sm.png)

    input size 4x4
    layer1 size 1x4 from input
    layer2 size 3x1 from layer1
    output size 1 from layer2

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-4x4-1x4-3x1-1-sm.png)

    input size 4x4
    output size 4x1 from input radius 0x2

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-4x4-4x1r0x2-sm.png)

    input size 16x16
    layer1 size 4x4 from input radius 1x1
    output size 7x1 from layer1

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-16x16-4x4r1x1-7-sm.png)

    # In the picture that follows, layerVertical is the set of 4 neurons
    # in the upper part of the picture, and layerHorizontal is the lower
    # set of 4 neurons.
    
    input size 6x6
    layerHorizontal size 2x2 from input radius 2x0
    layerVertical size 2x2 from input radius 0x2
    output size 1 from layerHorizontal
    output size 1 from layerVertical

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-6x6-2x2r2x0-2x2r0x2-1-sm.png)


    # This example shows how vertical and horizontal image features can be
    # extracted through separate paths and combined in a subsequent layer.

    input size 4x4

    layerH1 size 1x4 from input radius 4x0
    layerH2 size 1x4 from layerH1
    layerH3 size 1x4 from layerH2

    layerV1 size 4x1 from input radius 0x4
    layerV2 size 4x1 from layerV1
    layerV3 size 4x1 from layerV2

    output size 2 from layerV3
    output size 2 from layerH3


![](https://raw.github.com/davidrmiller/neural2d/master/images/net-4x4-hv-deep-sm.png)





How-do-I X?<a name="HowDoI"></a>
-------------

**How do I run the command-line program?**<a name="howConsole"></a>

Run neural2d with three arguments specifying the topology configuration,
input data configuration, and where to store the weights if training
succeeds:

     ./neural2d topology.txt inputData.txt weights.txt




**How do I run the GUI interface?**<a name="howGui"></a>

First launch the neural2d program with the -p option:

     ./neural2d topology.txt inputData.txt weights.txt -p

Then open a web browser and point it at
[http://localhost:24080](http://localhost:24080) .

If your firewall complains, you may need to allow access to TCP port
24080.




**How do I disable the GUI interface?**<a name="howDisableGui"></a>

Run CMake with the -DWEBSERVER=OFF option. Or if you are using
your own home-grown Makefiles, you can define the preprocessor
macro DISABLE_WEBSERVER. For example, with gnu compilers, add
-DDISABLE_WEBSERVER to the g++ command line. Alternatively, you can
undefine the macro ENABLE_WEBSERVER in neural2d.h.

When the web server is disabled, there is no remaining dependency on
POSIX sockets.

Also [see the illustrations above.](#GUI)




**How do I use my own data instead of the digits images?**<a name="howOwnData"></a>

For binary input data, create your own directory of BMP image
files or .dat binary files, and an input data config file
(inputData.txt) that follows the same format as shown in the
[examples elsewhere](#YourOwnData). Then define a [topology config
file](#topologyconfig) with the appropriate number of network inputs and
outputs, then run the neural2d program.

Or if you don't want to use .bmp image files or .dat binary files for
input, make an input config file containing all the literal input values
and the target output values.

See above for [more information on the input formats](#YourOwnData).




**How do I use a trained net on new data?**<a name="howTrained"></a>

It's all about the weights file. After the net has been successfully
trained, save the internal connection weights in a weights file. That's
typically done in neural2d.cpp by calling the member function
saveWeights(filename).

The weights you saved can be loaded back into a neural net of the same
topology using the member function loadWeights(filename). Once the net
has been loaded with weights, it can be used applied to new data by
calling feedForward(). Prior to calling feedForward(), you'll want to
set a couple of parameters:

     myNet.repeatInputSamples = false;
     myNet.reportEveryNth = 1;

This is normally done in neural2d.cpp.

You'll need to prepare a new input data config file (default name
inputData.txt) that contains a list of only those new input data images
that you want the net to process.




**How do I train on the MNIST handwritten digits data set?**<a name="MNIST"></a>

See the [instructions in the wiki](https://github.com/davidrmiller/neural2d/wiki/MNIST_Handwritten_dataset).




**How do I change the learning rate parameter?**<a name="howEta"></a>

In the command-line program, you can set the eta parameter or change it
by directly setting the eta member of the Net object, like this:

     myNet.eta = 0.1;

When using the web interface, you can change the eta parameter (and
other parameters) in the GUI at any time, even while the network is busy
processing input data.

Also see the [Parameter
List](https://github.com/davidrmiller/neural2d/wiki/ParameterList)
in the wiki.




**Are the output neurons binary or floating point?**<a name="howBinary"></a>

They are interpreted in whatever manner you train them to be, but you
can only train the outputs to take values in the range that the transfer
function is capable of producing.

If you're training a net to output binary values, it's best if you use
the maxima of the transfer function to represent the two binary values.
For example, when using the default tanh() transfer function, train
the outputs to be -1 and +1 for false and true. When using the logistic
transfer function, train the outputs to be 0 and 1.




**How do I use a different transfer function?**<a name="howTf"></a>

You can add a "tf" parameter to any layer definition line in
the topology config file.  The argument to tf can be "tanh",
"logistic", "linear", "ramp", "gaussian", or "relu".  The
transfer function you specify will be used by all the neurons
in that layer.  Here are the [graphs of the built-in transfer
functions.](https://github.com/davidrmiller/neural2d/wiki/TransferFunctions)

In the topology config file, the tf parameter is specified as in this
example:

     layerHidden1 size 64x64 from input radius 3x3 tf linear

You can add new transfer functions by following the examples in
neural2d-core.cpp.  There are two places to change: first find where
transferFunctionTanh() is defined and add your new transfer function and
its derivative there. Next, locate the constructor for class Neuron and
add a new else-if clause there, following the examples.




**How do I define a convolution filter?**<a name="howConvolve"></a>  

In the topology config file, any layer defined with a *convolve*
parameter and a list of constant weights will operate as a convolution
filter applied to the source layer.  The syntax is of the form:

     layer2 size 64x64 from input convolve {{1,0,-1},{0,0,0},{-1,0,1}}

Also [see above for more information](#ConvolutionFiltering).

See [this article](http://neural2d.net/?p=40) for the difference between 
*convolution filter* and *convolution networking*.




**How do I define convolution networking and pooling?**<a name="howConvolveNetworking"></a>  

In the topology config file, define a layer with an X,Y size and a depth
(number of kernels to train), and add a convolve parameter to specify
the kernel size. For example, to train 40 kernels of size 7x7 on an
input image of 64x64 pixels:

     input size 64x64  
     layerConv size 40*64x64 from input convolve 7x7
     . . .

To define a pooling layer, add a pool parameter, followed by the argument
"avg" or "max," followed by the operator size, e.g.:

     layerConv size 10*32x32 ...
     layerPool size 10*8x8 from layerConv pool max 4x4
     . . .

Also [see above for more information](#ConvolutionNetworking).




**How do the color image pixels get converted to floating point for the input layer?**<a name="howRgb"></a>

That's in the ImageReaderBMP class in neural2d-core.cpp. The default
version provided converts each RGB pixel to a single floating point
value in the range -1.0 to 1.0.

By default, the three color channels are converted to monochrome and
normalized to the range -1.0 to 1.0. That can be changed at runtime by
setting the colorChannel member of the Net object to R, G, B, or BW prior
to calling feedForward(). E.g., to use only the green color channel of
the images, use:

    myNet.colorChannel = NNet::G;

The color conversion can also be specified in the topology config file on
the line that defines the input layer by setting the "channel" parameter
to R, G, B, or BW, e.g.:

    input size 64x64 channel G

There is no conversion when inputting floating point data directly from a
.dat file, or from literal values embedded in the input data config file.




**How can I use .jpg and .png images as inputs to the net?**<a name="howJpg"></a>

Currently only .bmp images files are supported. This is because the
uncompressed BMP format is so simple that we can use simple, standard
C/C++ to read the image data without any dependencies on third-party
image libraries.

To support a new input file format, derive a new subclass
from class ImageReader and implement its getData() member
following the examples of the existing image readers. For
more information on the input data readers, see the [design
notes.](#https://github.com/davidrmiller/neural2d/issues/21)





**How can I find my way through the source code?**<a name="sourcecode"></a>

Here is a little map of the important files:

![](https://raw.github.com/davidrmiller/neural2d/master/file-relationships.png)

Also see the class relationship diagram in the project root directory.





**Why does the net error rate stay high? Why doesn't my net learn?**<a name="howLearn"></a>

Neural nets are finicky. Try different network topologies. Try starting
with a larger eta values and reduce it incrementally. It could also
be due to redundancy in the input data, or mislabeled target output
values. Or you may need more training samples.




**What other parameters do I need to know about?**<a name="howParams"></a>

Check out the [list of parameters in the
wiki](https://github.com/davidrmiller/neural2d/wiki/ParameterList).





Licenses<a name="Licenses"></a>
--------

The neural2d program and its documentation are
copyrighted and licensed under the terms of the [MIT
license](http://opensource.org/licenses/MIT). See the LICENSE file for
more information.

The set of digits images in the images/digits/ subdirectory is released
to the public domain.
