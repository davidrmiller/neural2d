Neural2d - Neural Net Simulator
================================

User Manual
===========

Ver. 1.0  
Updated 14-May-2015

Intro video (11 min): [https://www.youtube.com/watch?v=yB43jj-wv8Q](https://www.youtube.com/watch?v=yB43jj-wv8Q)

Features
--------

*     Optimized for 2D image data -- input data can be read from .bmp image files
*     Neuron layers can be abstracted as 1D or 2D arrangements of neurons
*     Network topology is defined in a text file
*     Neurons in layers can be fully or sparsely connected
*     Selectable transfer function per layer
*     Adjustable or automatic training rate (eta)
*     Optional momentum (alpha) and regularization (lambda)
*     Convolution filtering and convolution networking
*     Standalone console program
*     Simple, heavily-commented code, suitable for prototyping, learning, and experimentation
*     Optional web-browser-based GUI controller
*     No dependencies! Just C++11 (and POSIX networking for the optional webserver interface)

Document Contents
-----------------

[Requirements](#Requirements)  
[How to run the digits demo](#Demo)  
[How to run the XOR example](#XorExample)  
[GUI interface](#GUI)  
[How to use your own data](#YourOwnData)  
[The 2D in neural2d](#2D)  
[Convolution filtering](#ConvolutionFiltering)  
[Convolution networking and pooling](#ConvolutionNetworking)  
[Topology config file format](#TopologyConfig)  
[Topology config file examples](#TopologyExamples)  
[How-do-I *X*?](#HowDoI)  
* [How do I get, build, and install the command-line neural2d program?](#howInstall)  
* [How do I run the command-line program?](#howConsole)  
* [How do I run the GUI interface?](#howGui)  
* [How do I disable the GUI interface?](#howDisableGui)  
* [How do I use my own data instead of the digits images?](#howOwnData)  
* [How do I use a trained net on new data?](#howTrained)  
* [How do I train on the MNIST handlwritten digits data set?](#MNIST)  
* [How do I change the learning rate parameter?](#howEta)  
* [Are the output neurons binary or floating point?](#howBinary)  
* [How do I use a different transfer function?](#howTf)  
* [How do I define a convolution filter?](#howConvolve)  
* [How do I define convolution networking and pooling?](#howConvolveNetworking)  
* [How do the color image pixels get converted to floating point for the input layer?](#howRgb)  
* [How can I use .jpg and .png images as inputs to the net?](#howJpg)  
* [Why does the net error rate stay high? Why doesn't my net learn?](#howLearn)  
* [What other parameters do I need to know about?](#howParams)  

[Licenses](#Licenses)  

Also see the [wiki](https://github.com/davidrmiller/neural2d/wiki) for more information.


Requirements<a name="Requirements"></a>
------------

* C++-11 compiler (e.g., g++ on Linux)
* POSIX sockets (e.g., Cygwin on Windows)
* Compiles and runs on Linux, Windows, and probably Mac


How to run the digits demo<a name="Demo"></a>
-----------

Place all the files, maintaining the relative directory structure, into a convenient directory.

In the images/digits/ subdirectory, extract the image files from the archive into the same directory.

To compile neural2d, cd to the directory containing the Makefile and execute the default make target:

    make

This will use g++ to compile neural2d.cpp and neural2d-core.cpp and result in an executable
named neural2d.

To run the demo, execute:

    make test

In this demo, we train the neural net to recognize digits. The input data, or "training set",
consists of a few thousand images of numeric digits. The first 50 look like these:

![training samples](https://raw.github.com/davidrmiller/neural2d/master/images/digits-illus.png)

The images are 32x32 pixels each, stored in .bmp format. In this demo, the neural net is 
configured to have 32x32 input neurons, and 10 output neurons. The net is trained to classify 
the digits in the images and to indicate the answer by driving the corresponding output 
neuron to a high level.

Once the net is sufficiently trained, all the connection weights are saved in a file
named "weights.txt".



How to run the XOR example<a name="XorExample"></a>
--------------------------

In the top level directory where the Makefile lives, execute:

     make test-xor

For more information about the XOR example, see
[this wiki page](https://github.com/davidrmiller/neural2d/wiki/XOR_Example).



GUI interface (optional)<a name="GUI"></a>
-------------

First, launch the neural2d console program in a command window with the -p option:

     ./neural2d topology.txt inputData.txt weights.txt -p

The -p option causes the neural2d program to wait for a command before starting
the training. The screen will look something like this:

![console-window](https://raw.github.com/davidrmiller/neural2d/master/images/console1.png)

At this point, the neural2d console program is paused and waiting for a command
to continue. Using any web browser, open:

     http://localhost:24080

A GUI interface will appear that looks like:

![HTTP interface](https://raw.github.com/davidrmiller/neural2d/master/images/gui2-sm.png)

Press Resume to start the neural net training. It will automatically pause when the
average error rate falls below a certain threshold (or when you press Pause). You 
now have a trained net. You can press Save Weights to save the weights for later use.


How to use your own data<a name="YourOwnData"></a>
------------------------

If you are inputting data from image files, you'll need to prepare a set of BMP image files
and an input data config file. The config file (named inputData.txt by default) is a list
of image filenames to use as inputs to the neural net, and optionally the target output
values for each image. The format looks like this example:

    images/thumbnails/test-918.bmp -1 1 -1 -1 -1 -1 -1 -1 -1 -1
    images/thumbnails/test-919.bmp -1 -1 -1 -1 -1 -1 -1 -1 1 -1
    images/thumbnails/test-920.bmp -1 -1 -1 -1 -1 -1 1 -1 -1 -1
    images/thumbnails/test-921.bmp -1 -1 -1 -1 -1 1 -1 -1 -1 -1

The path and filename cannot contain any spaces.

If you are not using image files for input, you'll need to prepare an input config file
(named inputData.txt by default) similar to the above but with the literal input values
inside curly braces. For example, for a net with eight inputs and two outputs, the format
is like this:

     { 0.32 0.98 0.12 0.44 0.98 1.2 1 -1 } -1 1 

You'll also need a topology config file (named topology.txt by default). It
contains a specification of the neural net topology (the number and arrangement of neurons 
and connections). Its format is described in a later section. A typical one looks
something like this:

    input size 32x32  
    layer1 size 32x32 from input radius 8x8  
    layer2 size 16x16 from layer1  
    output size 1x10 from layer2  

Then run neuron2d (optionally with the web browser interface) and experiment with the parameters 
until the net is adequately trained, then save the weights in a file for later use.

If you run the web interface, you can change the global parameters from the GUI while the
neural2d program is running. If you run the neural2d console program without the GUI interface,
there is no way to interact with it while running. Instead, you'll need to examine and
modify the parameters in the code at the top of the files neural2d.cpp and neural2d-core.cpp.


The 2D in neural2d<a name="2D"></a>
------------------

In a simple traditional neural net model, the neurons are arranged in a column in each layer:

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-542-sm.png)

In neural2d, you can specify a rectangular arrangement of neurons in each layer, such as:

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-5x5-4x4-2-sm.png)

The neurons can be sparsely connected to mimic how retinal neurons are connected in 
biological brains. For example, if a radius of "1x1" is specified in the topology 
config file, each neuron on the right (destination) layer will connect to a circular patch
of neurons in the left (source) layer as shown here (only a single neuron on the right 
side is shown connected in this picture so you can see what's going on, but imagine 
all of them connected in the same pattern):

![radius-1x1](https://raw.github.com/davidrmiller/neural2d/master/images/proj-1x1-sm.png )

The pattern that is projected onto the source layer is elliptical. (Layers configured as
convolution filters work slightly differently; see the later section about convolution filtering.)

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

Any layer other than the input layer can be configured as a convolution filter layer by
specifying a *convolve-matrix* specification for the layer in the topology config file.
The neurons are still called neurons, but their operation differs in the following ways:

* The connection pattern to the source layer is defined by the convolution matrix (kernel) dimensions
(not by a *radius* parameter)

* The connection weights are initialized from the convolution matrix, and are
constant throughout the life of the net.

* The transfer function is automatically set to the identity function (not by a *tf* parameter).

For example, the following line in the topology config file defines a 3x3 convolution matrix
for shrarpening the source layer:

     layerConv1 size 64x64 from input convolve {{0,-1,0},{-1,5,-1},{0,-1,0}}

When a convolution matrix is specified for a layer, you cannot also specify a *radius* 
parameter for that layer, as the convolution matrix size determines the size and shape of 
the rectangle of neurons in the source layer. You also cannot also specify a *tf* parameter,
because the transfer function on a convolution layer is automatically set to be the 
identity function.

The elements of the convolution matrix are stored as connection weights to the source
neurons. Connection weights on convolution layers are not updated by the back propagation
algorithm, so they remain constant for the life of the net.

The results are undefined if a layer is defined as both a convolution layer and a
regular layer.

For illustrations of various convolution kernels, see
[this Wikipedia article](http://en.wikipedia.org/wiki/Kernel_%28image_processing%29)

In the following illustration, the topology config file defines a convolution filter with a
2x2 kernel that is applied to the input layer, then the results are combined
with a reduced-resolution fully-connected pathway. The blue connections in the picture are the
convolution connections; the green connections are regular neural connections:

    input size 8x8
    layerConvolve size 8x8 from input convolve {{-1,2},{-1,2}}
    layerReducedRes size 4x4 from input
    output size 2 from layerConvolve
    output size 2 from layerReducedRes

![](https://raw.github.com/davidrmiller/neural2d/master/images/net-convolve-8x8.png)



Convolution networking and pooling<a name="ConvolutionNetworking"></a>
---------------------

A **[convolution network layer](http://en.wikipedia.org/wiki/Convolutional_neural_network)** is like 
a convolution filter layer, except that the kernel participates in backprop training, and everything 
inside the layer is replicated *N* times to train *N* separate kernels. A convolution network layer 
is said to have depth *N*. A convolution network layer has *depth* \* *X* \* *Y* neurons.

Any layer other than the input or output layer can be configured as a convolution networking layer by 
specifying a layer size with a depth > 1, and specifying the kernel size with a convolve parameter. 
For example, to train 40 kernels of size 7x7 on an input image of 64x64 pixels:

      input size 64x64
      layerConv size 40*64x64 from input convolve 7x7
      . . .

A **[pooling layer](http://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer)** 
down-samples the previous layer by finding the average or maximum in patches of source neurons. 
A pooling layer is defined in the topology config file by specifying a pool parameter on a layer. 
Pooling layers can take their input from any other kind of layer of equal depth or from a
regular layer of depth 1.

In the topology config syntax, the pool parameter requires the argument "avg" or "max" followed by 
the operator size, For example, in a convolution network pipeline of depth 20, you might have 
these layers:

      input size 64x64
      layerConv size 20*64x64 from input convolve 5x5
      layerPool size 20*16x16 from layerConv pool max 4x4
      . . .

If a convolution network or pooling layer is fed into a regular layer of depth 1, it will be
fully connected.


Topology config file format<a name="TopologyConfig"></a>
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

1. If a size parameter is omitted, the size is copied from the layer specified in the from parameter.

1. A radius parameter cannot be used on the same line with a convolve or pool parameter.

1. Pooling layers and convolution networking layers (which have depth > 1) can feed into a 
pooling or convolution layer of the same depth, or into a regular layer. A pooling layer 
fed into a regular layer will be fully connected to all the neurons in the entire depth of
the layer, or sparsely 2D-connected at all depths if a radius parameter is specified.

1. The same layer name can be defined multiple times with different "from" parameters.
This allows source neurons from more than one layer to be combined in one 
destination layer. The source layers can be any size. When a destination layer is 
defined more than once, the first definition must have a *size* parameter. The size parameter
is optional on the repeated lines; if it appears, it must be the same size as defined
initially. For example, in the following, layerCombined is size 8x8:

     input size 128x128  
     layerVertical size 32x32 from input radius 1x8  
     layerHorizontal size 16x16 from input radius 8x1  
     **layerCombined** **size 8x8** from layerVertical   
     **layerCombined** from layerHorizontal  
     output size 1 from layerCombined  

1. In the *xy-spec*  and in the X,Y part of the *dxy-spec*, you may specify one or two dimensions.
Spaces are not allowed in the size spec. If only one dimension is given, the other is assumed to be 1. 
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

**How do I get, build, and install the command-line neural2d program?**<a name="howInstall"></a>

Get the files from:

    https://github.com/davidrmiller/neural2d

Put those into a directory. Expand the images files in the images/digits/ subdirectory. In the top level
directory (where the Makefile lives), build the program with:

     make

That will produce an executable named neuron2d in the same directory.

To test the installation, run:

     make test

If it succeeds, it will create a weights.txt file of non-zero size.

**How do I run the command-line program?**<a name="howConsole"></a>

     ./neural2d topology.txt inputData.txt weights.txt

**How do I run the GUI interface?**<a name="howGui"></a>

First launch the neural2d program with the -p option:

     ./neural2d topology.txt inputData.txt weights.txt -p

Then open a web browser and point it at [http://localhost:24080](http://localhost:24080) .

If your firewall complains, you may need to allow access to TCP port 24080.

**How do I disable the GUI interface?**<a name="howDisableGui"></a>

The easiest way is to add -DDISABLE_WEBSERVER to the g++ command line in the Makefile.
Alternatively, you can undefine the macro ENABLE_WEBSERVER in neural2d.h.

When the web server is disabled, there is no remaining dependency on POSIX sockets.

**How do I use my own data instead of the digits images?**<a name="howOwnData"></a>

Create your own directory of BMP images, and a config file that follows the same format as
shown in the provided default inputData.txt. Then define a topology config file with the
appropriate number of network inputs and outputs, then run the neural2d program.

Or if you don't want to use image files for input, make an input config file containing
all the literal input values and the target output values. The format is described
in an earlier section.

**How do I use a trained net on new data?**<a name="howTrained"></a>

It's all about the weights file. After the net has been successfully trained, save 
the internal connection weights in a weights file.
That's typically done in neural2d.cpp by calling the member function saveWeights(filename).

The weights you saved can be loaded back into a neural net of the same topology using
the member function loadWeights(filename). Once the net has been loaded with weights,
it can be used applied to new data by calling feedForward(). Prior to calling
feedForward(), you'll want to set a couple of parameters:

     myNet.repeatInputSamples = false;
     myNet.reportEveryNth = 1;

This is normally done in neural2d.cpp.

You'll need to prepare a new input data config file (default name inputData.txt)
that contains a list of only those new input data images that you want the net to
process.

**How do I train on the MNIST handlwritten digits data set?**<a name="MNIST"></a>

See the [instructions in the wiki](https://github.com/davidrmiller/neural2d/wiki/MNIST_Handwritten_dataset).

**How do I change the learning rate parameter?**<a name="howEta"></a>

In the command-line program, you can set the eta parameter or change it by directly
setting the eta member of the Net object, like this:

     myNet.eta = 0.1;

When using the web interface, you can change the eta parameter (and other parameters)
in the GUI at any time, even while the network is busy processing input data.

Also see the [Parameter List](https://github.com/davidrmiller/neural2d/wiki/ParameterList)
in the wiki.

**Are the output neurons binary or floating point?**<a name="howBinary"></a>

They are interpreted in whatever manner you train them to be, but you can only 
train the outputs to take values in the range that the transfer function is 
capable of producing.

If you're training a net to output binary values, it's best if you use the 
maxima of the transfer function to represent the two binary values.
For example, when using the default tanh() transfer function, train the outputs to
be -1 and +1 for false and true. When using the logistic transfer function, train the
outputs to be 0 and 1.

**How do I use a different transfer function?**<a name="howTf"></a>

You can add a "tf" parameter to any layer definition line in the topology config file.
The argument to tf can be "tanh", "logistic", "linear", "ramp", "gaussian", or "relu". 
The transfer function you specify will be used by all the neurons in that layer.
See neural2d-core.cpp for more information.

In the topology config file, the tf parameter
is specified as in this example:

     layerHidden1 size 64x64 from input radius 3x3 tf linear

You can add new transfer functions by following the examples in neural2d-core.cpp.
There are two places to change: first find where transferFunctionTanh() is defined
and add your new transfer function and its derivative there. Next, locate the constructor
for class Neuron and add a new else-if clause there, following the examples.

**How do I define a convolution filter?**<a name="howConvolve"></a>  

In the topology config file, any layer defined with a *convolve* parameter and a list
of constant weights will operate as a convolution filter applied to the source layer. 
The syntax is of the form:

     layer2 size 64x64 from input convolve {{1,0,-1},{0,0,0},{-1,0,1}}

**How do I define convolution networking and pooling?**<a name="howConvolveNetworking"></a>  

In the topology config file, define a layer with an X,Y size and a depth (number of kernels
to train), and add a convolve parameter to specify the kernel size. For example, to train 40 
kernels of size 7x7 on an input image of 64x64 pixels:

     input size 64x64  
     layerConv size 40*64x64 from input convolve 7x7
     . . .

To define a pooling layer, add a pool parameter, followed by the argument "avg" or "max,"
followed by the operator size, e.g.:

     layerConv size 10*32x32 ...
     layerPool size 10*8x8 from layerConv pool max 4x4
     . . .

**How do the color image pixels get converted to floating point for the input layer?**<a name="howRgb"></a>

That's in the ReadBMP() function in neural2d-core.cpp. The default version of ReadBMP()
converts each RGB pixel to a single floating point value in the range 0.0 to 1.0.

By default, the RGB color pixels are converted to monochrome and normalized to the
range 0.0 to 1.0. That can be changed at runtime by setting the colorChannel
member of the Net object to R, G, B, or BW prior to calling feedForward().
E.g., to use only the green color channel of the images, use:

    myNet.colorChannel = NNet::G;

The color conversion can also be specified in the topology config file on the
line that defines the input layer by setting the "channel" parameter to R, G, B, or BW, e.g.:

    input size 64x64 channel G

**How can I use .jpg and .png images as inputs to the net?**<a name="howJpg"></a>

Currently only .bmp images are supported. This is because the uncompressed BMP format is so simple
that we can use simple, standard C/C++ to read the image data without any dependencies on third-party
image libraries. To add an adapter for other image formats, follow the example of the
ReadBMP() function and write a new adapter such as ReadJPG(), ReadPNG(), etc., using your
favorite image library, then replace the call to ReadBMP() with your new function.

**Why does the net error rate stay high? Why doesn't my net learn?**<a name="howLearn"></a>

Neural nets are finicky. Try different network topologies. Try starting with a larger
eta values and reduce it incrementally. It could also be due to redundancy in the input data, 
or mislabeled target output values. Or you may need more training samples.

**What other parameters do I need to know about?**<a name="howParams"></a>

Check out the [list of parameters in the wiki](https://github.com/davidrmiller/neural2d/wiki/ParameterList).


Licenses<a name="Licenses"></a>
--------

The neural2d program and its documentation are copyrighted and licensed under the terms of the MIT license.

The set of digits images in the images/digits/ subdirectory is released to the public domain.
