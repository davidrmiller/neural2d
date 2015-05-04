/*
neural2d.h
David R. Miller, 2015
See https://github.com/davidrmiller/neural2d for more information.
*/

/*
 * Notes:
 *
 * This is a backpropagation neural net simulator with these features:
 *
 *    1. Optimized for 2D image data -- input data is read from .bmp image files
 *    2. Neuron layers can be abstracted as 1D or 2D arrangements of neurons
 *    3. Network topology is defined in a text file
 *    4. Neurons in layers can be fully or sparsely connected
 *    5. Selectable transfer function per layer
 *    6. Adjustable or automatic training rate (eta)
 *    7. Optional momentum (alpha) and regularization (lambda)
 *    8. Standalone console program
 *    9. Heavily-commented code, < 3000 lines, suitable for prototyping, learning, and experimentation
 *   10. Optional web GUI controller
 *
 * This program is written in the C++-11 dialect. It uses mostly ISO-standard C++
 * features and a few POSIX features that should be widely available on any
 * compiler that knows about C++11 and POSIX. Other than that, our goal is
 * to avoid any dependencies on any third-party library.
 *
 * This is a console program that requires no GUI. However, a GUI is optional:
 * The WebServer class provides an HTTP server that a web browser can connect to
 * at port 24080.
 *
 * For most array indices and offsets into containers, we use 32-bit integers
 * (uint32_t).
 *
 * For training input samples, we use BMP images because we can read those
 * easily with standard C/C++ without using image libraries. The function ReadBMP()
 * reads an image file and converts the pixel data into an array (container) of
 * floats. When an image is first read, we'll cache the image data in memory to
 * make subsequent reads faster (because we may want to input the training samples
 * multiple times during training).
 *
 * There are three conceptual modes of operation, or use-cases:
 *
 *     TRAINING: input samples are labeled with target output values, and
 *               weights are adjusted during training. For this mode,
 *               call feedForward(), backProp(), and reportResults()
 *               once per input sample. Call saveWeights() once the
 *               net is trained.
 *
 *     VALIDATE: input samples are labeled with target output values,
 *               output values are reported, but weights are NOT adjusted.
 *               For this mode, call loadWeights(), then call
 *               feedForward() and reportResults() once per input sample.
 *
 *     TRAINED:  input samples have no target output values; outputs are
 *               reported, weights are NOT adjusted. For this mode,
 *               call loadWeights(), then call feedForward() and
 *               reportResults() once per input sample.
 *
 * Typical operation is to label a bunch of input samples, use some of them in
 * TRAINING mode, and save the weights when the net is trained. Then using the saved
 * weights and some more labeled training samples, test how well the net performs
 * in VALIDATE mode. If successful, the saved weights can be used in TRAINED
 * mode.
 *
 * Class relationships: Everything is in the NNet namespace. Class Net can be
 * instantiated to create a neural net. The Net object holds a container of
 * Layer objects. Each Layer object holds a container of Neuron objects. Each
 * Neuron object holds containers of references to Connection objects which
 * define the connections. The container of Connection objects is held in the
 * net object and neurons refer to connections by indices. Class SampleSet holds
 * the input samples that are presented to the neural net when the
 * feedForward() member is called.
 *
 * There are three kinds of layers of neurons: regular layers, convolution filter
 * layers, and convolution network layers. All layers have a depth. Regular layers
 * and convolution filter layers have a depth of 1. Convolution network layers
 * have a depth > 1.
 */

#ifndef NNET_H
#define NNET_H

// The embedded webserver GUI can be disabled either by undefining ENABLE_WEBSERVER
// here, or by defining DISABLE_WEBSERVER (e.g., on the g++ command line, add
// -DDISABLE_WEBSERVER. I.e., to enable the webserver, ENABLE_WEBSERVER must be
// defined and DISABLE_WEBSERVER must not be defined.

#define ENABLE_WEBSERVER


// ISO-standard C++ headers:

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    #include <condition_variable> // For mutex
    #include <thread>
    #include <sys/socket.h>  // POSIX sockets
    #include <netinet/in.h>  // POSIX sockets
    #include "webserver.h"
#endif


// Everything we define in this file will be inside the NNet namespace. This keeps
// all of our definitions out of the global namespace. (You can indent all the source
// lines inside the namespace below if you're an indenting purist.)
//
namespace NNet {

// For convenience, we'll bring a few common std:: names into the NNet namespace:
using std::endl;
using std::string;
using std::vector;

// On Windows, cygwin is missing to_string(), so we'll make one here:
#if defined(__CYGWIN__)
    #include <sstream>
    template <typename T>
    std::string to_string(T value)
    {
      std::ostringstream os;
      os << value;
      return os.str();
    }
#else
  using std::to_string;
#endif


// In the NNet namespace, we use Logger objects for all console output.
// These are named info, warn, and err. By default, they are mapped to
// std::cout and std::cerr. The Logger class provides a centralized place to
// capture, filter, or redirect program output if needed.
//
class Logger
{
public:
    std::ostream *pfile;
    Logger(std::ostream &o = std::cout) : pfile(&o) { };

    template <typename T>
    Logger& operator<<(const T &val) {
        *pfile << val;
        return *this;
    }

    Logger& operator<<(std::ostream& (*pfunc)(std::ostream &)) {
        *pfile << pfunc;
        return *this;
    }
};

// These loggers can be used like cout and cerr:
extern Logger info, warn, err;


// When we throw exceptions, they will be one of the following:
class exceptionConfigFile       : public std::exception { };
class exceptionImageFile        : public std::exception { };
class exceptionInputSamplesFile : public std::exception { };
class exceptionWeightsFile      : public std::exception { };
class exceptionRuntime          : public std::exception { };


enum ColorChannel_t { R, G, B, BW };
enum poolMethod_t { POOL_NONE, POOL_MAX, POOL_AVG };


// One Sample holds one set of neural net input values, and the expected output
// values (if known in advance). If the input data was extracted from a 2D image,
// the pixel values are stored flattened (linearized) in member .data.
//
class Sample
{
public:
    string imageFilename;
    vector<float> &getData(ColorChannel_t colorChannel);
    void clearCache(void);
    vector<float> targetVals;
    vector<float> data; // Pixel data converted to floats and flattened to a 1D array
};


// A SampleSet object holds a container of all the input samples to be processed
// by the neural net:
//
class SampleSet
{
public:
    SampleSet() { samples.clear(); };
    void loadSamples(const string &inputDataConfigFilename);
    void shuffle(void);
    void clearCache(void);
    void clearImageCache(void); // Clear only cached image data; retain explicit inputs values
    vector<Sample> samples;
};


// Little structure to hold a value consisting of a depth and X and Y:
struct dxySize {
    uint32_t depth;  // Only convolution network layers have nonzero depth
    uint32_t x;
    uint32_t y;
};

// Little structure to hold various radii in units of neurons:
struct xySize {
    uint32_t x;
    uint32_t y;
};


class Neuron;     // Forward references
class Connection;


typedef vector<float> matColumn_t;
typedef vector<matColumn_t> convolveMatrix_t; // Allows access as convolveMatrix[x][y]

typedef float (*transferFunction_t)(float); // Also used for the derivative function


// Layer-specific meta-information:
//
struct layerParams_t {
    layerParams_t(void);
    void resolveTransferFunctionName(void);
    void clear(void);

    string layerName;                  // Can be input, output, or layer*
    bool isConvolutionFilterLayer;     // Equivalent to (convolveMatrix.size() == 1)
    bool isConvolutionNetworkLayer;    // Equivalent to (convolveMatrix.size() > 1)
    dxySize size;                      // layer depth, X, Y dimensions
    ColorChannel_t channel;            // Applies only to the input layer
    xySize radius;                     // Always used, so set high to fully connect layers
    string transferFunctionName;
    transferFunction_t tf;
    transferFunction_t tfDerivative;
    xySize kernelSize;                 // Used only for convolution network layers
    vector<convolveMatrix_t> convolveMatrix;   // zero, one, or multiple kernels depending on layer type
    enum poolMethod_t poolMethod;      // Used only for convolution network layers
    xySize poolSize;                   // Used only for convolution network layers
};


// This structure holds the information extracted from a single line in
// the topology config file. The topology file parser creates one of these
// objects for each line in the config file. The parser fills in as much
// of the layerParams fields as it can. Net::parseConfigFile() is responsible
// for converting a container of topologyConfigSpec_t into layers of neurons.
//
struct topologyConfigSpec_t {
    topologyConfigSpec_t(void);
    unsigned configLineNum;            // Used for reporting errors in topology config file
    string fromLayerName;              // Can be any existing layer name
    size_t fromLayerIndex;             // Index into the Net::layers container
    bool sizeSpecified;
    bool colorChannelSpecified;
    bool radiusSpecified;
    bool tfSpecified;
    layerParams_t layerParams;
};


//  ***********************************  class Layer  ***********************************

// Each layer is conceptually a bag of neurons in a 2D arrangement, stored
// flattened in a 1D container:
//
struct Layer
{
    layerParams_t params;
    vector<Connection> *pConnections;   // Pointer to the container of all Connection records

    // For each layer, before any references are made to its members, the .neurons
    // member must be initialized with sufficient capacity to prevent reallocation.
    // This allows us to form stable pointers, iterators, or references to neurons.
    vector<Neuron> neurons;  // 3D array, flattened index = depth*(sizeX*sizeY) + y * sizeX + x
};


// ***********************************  class Connection  ***********************************

// If neurons are considered as nodes in a directed graph, the edges would be the
// "connections". Each connection goes from one neuron to any other neuron in any
// layer. Connections are analogous to synapses in natural biology. The .weight
// member is what it's all about -- once a net is trained, we only need to save
// the weights of all the connections. The set of all weights plus the network
// topology defines the neural net's function. The .deltaWeight member is used
// only for the momentum calculation.
//
struct Connection
{
    Connection(Neuron &from, Neuron &to);
    Neuron &fromNeuron;
    Neuron &toNeuron;
    float weight;
    float deltaWeight;  // The weight change from the previous training iteration
};


// ***********************************  class Neuron  ***********************************

class Neuron
{
public:
    Neuron();
    Neuron(const Layer *pMyLayer);
    float output;
    float gradient;

    // All the input and output connections for this neuron. We store these as indices
    // into an array of Connection objects stored somewhere else. We store indices
    // instead of pointers or iterators because the container of Connection objects
    // can get reallocated which would invalidate the references. Currently the connections
    // container is a public data member of class Net, but it could be stored anywhere that
    // is accessible.

    vector<uint32_t> backConnectionsIndices; // My back connections
    vector<uint32_t> forwardConnectionsIndices; // My forward connections

    void feedForward(void);                     // Propagate the net inputs to the outputs
    void updateInputWeights(float eta, float alpha);  // For backprop training
    void calcOutputGradients(float targetVal);        // For backprop training
    void calcHiddenGradients(void);                   // For backprop training

    // The only reason for the .sourceNeurons member is to make it easy to
    // find and report any unconnected neurons. For everything else, we'll use
    // the backConnectionIndices to find the neuron's inputs. This container
    // holds pointers to all the source neurons that feed this neuron:

    std::set<Neuron *> sourceNeurons;

private:
    const Layer *pMyLayer;  // Used to get the transfer function pointers
    float sumDOW_nextLayer(void) const;        // Used in hidden layer backprop training
};


// ***********************************  class Net  ***********************************

class Net
{
public: // This section exposes the complete public API for class Net

    // Parameters that affect overall network operation. These can be set by
    // directly accessing the data members:

    bool enableBackPropTraining; // If false, backProp() won't update any weights
    float doneErrorThreshold;    // Pause when overall avg error falls below this
    float eta;                   // Initial overall net learning rate, [0.0..1.0]
    bool dynamicEtaAdjust;       // true enables automatic eta adjustment during training
    float alpha;                 // Initial momentum, multiplier of last deltaWeight, [0.0..1.0]
    float lambda;                // Regularization parameter. If zero, regularization is disabled:
    string weightsFilename;      // Filename to use in saveWeights() and loadWeights()
    float error;                 // Overall net error
    float recentAverageError;    // Averaged over recentAverageSmoothingFactor samples

    // When a net topology specifies sparse connections (i.e., when there is a radius
    // parameter specified in the topology config file), then the shape of the area
    // that is back-projected onto the source layer of neurons can be elliptical or
    // rectangular. The default is elliptical (false).
    bool projectRectangular;

    bool isRunning;     // If true, start processing without waiting for a "resume" command

    // To reduce screen clutter during training, reportEveryNth can be set > 1. When
    // in VALIDATE or TRAINED mode, you'll want to set this to 1 so you can see every
    // result:
    uint32_t reportEveryNth;

    // For some calculations, we use a running average of net error, averaged over
    // this many input samples:
    float recentAverageSmoothingFactor;

    // If repeatInputSamples is false, the program will pause after running all the
    // input samples once. If set to true, the input samples will automatically repeat.
    // If shuffleInputSamples is true, then the input samples will be randomly
    // shuffled after each use:
    bool repeatInputSamples;
    bool shuffleInputSamples;


    Net(const string &topologyFilename);          // ctor
    ~Net(void);

    void feedForward(void);                       // Propagate inputs to outputs
    void feedForward(Sample &sample);
    void backProp(const Sample &sample);          // Backprop and update all weights

    // The connection weights can be saved or restored at any time. Note that the network
    // topology is not saved in the weights file, so you'll have to manually keep track of
    // which weights file goes with which topology file.
    bool saveWeights(const string &filename) const;
    bool loadWeights(const string &filename);

    // Functions useful after forward propagation:
    float getNetError(void) const { return error; };
    float getRecentAverageError(void) const { return recentAverageError; };
    void calculateOverallNetError(const Sample &sample);  // Update .error member

    // Functions for displaying the results when processing input samples:
    void reportResults(const Sample &sample) const;
    void debugShowNet(bool details = false);      // Display details of net topology

public: // These members are public only for convenience of unit testing.
    uint32_t inputSampleNumber; // Increments each time feedForward() is called
    SampleSet sampleSet;     // List of input images and access to their data
    static const uint32_t HUGE_RADIUS = 1e9; // Magic value

    // Here is where we store all the weighted connections. The container can get
    // reallocated, so we'll only refer to elements by indices, not by pointers or
    // references. This also allows us to hack on the connections container during
    // training if we want to, by dynamically adding or deleting connections without
    // invalidating references.

    vector<Connection> connections;
    vector<Layer> layers;
    Neuron bias;  // Fake neuron with constant output 1.0
    float lastRecentAverageError;    // Used for dynamically adjusting eta
    uint32_t totalNumberConnections; // Including 1 bias connection per neuron
    uint32_t totalNumberNeurons;
    vector<topologyConfigSpec_t> parseTopologyConfig(std::istream &cfg);
    void reportUnconnectedNeurons(void);
    float adjustedEta(void);
    void configureNetwork(vector<topologyConfigSpec_t> configSpecs, const string configFilename = "");

private:
    void parseConfigFile(const string &configFilename); // Creates layer metadata from a config file
    convolveMatrix_t parseMatrixSpec(std::istringstream &ss);
    Layer &createLayer(const layerParams_t &params);
    bool addToLayer(Layer &layerTo, Layer &layerFrom);
    void createNeurons(Layer &layerTo, Layer &layerFrom);
    void connectNeuron(Layer &layerTo, Layer &layerFrom, Neuron &neuron,
                       uint32_t nx, uint32_t ny);
    void connectBias(Neuron &neuron);
    int32_t getLayerNumberFromName(const string &name) const;

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    void doCommand(); // Handles incoming program command and control
    void actOnMessageReceived(Message_t &msg);
    void makeParameterBlock(string &s);
    WebServer webServer;
    int portNumber;
    MessageQueue messages;
#endif
};

extern uint32_t flattenDXY(uint32_t depth, uint32_t x, uint32_t y, uint32_t xSize, uint32_t ySize);
extern uint32_t flattenDXY(uint32_t depth, uint32_t x, uint32_t y, dxySize size);
extern uint32_t flattenDXY(dxySize coord3d, xySize size);


} // end namespace NNet

#endif // end #ifndef NNET_H
