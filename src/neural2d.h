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
 *   11. Convolution filtering
 *   12. Convolution networking
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
 * There are four kinds of layers of neurons: regular layers, convolution filter
 * layers, convolution network layers, and pooling layers. All layers have a depth:
 * regular layers and convolution filter layers have a depth of 1; convolution
 * network layers and pooling layers have a depth > 1 (where depth is the number
 * of convolution kernels to train).
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
#include <memory>   // for unique_ptr
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
// all of our definitions out of the global namespace.
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


enum ColorChannel_t { COLOR_NONE, R, G, B, BW };
enum poolMethod_t { POOL_NONE, POOL_MAX, POOL_AVG };

float pixelToNetworkInputRange(unsigned val);  // Converts uint8_t to float


// ImageReader objects are used to read various image file formats and extract the
// image data. A subclass of ImageReader must be defined for each supported file
// format. When .getData() is called, the image reader will attempt to read
// the image data in the file. If successful, the image data is converted to
// floating point, and saved in dataContainer as a flattened 1D array. If successful,
// the function returns the nonzero image size. If the image reader cannot read
// the image data for any reason, it will silently return a size of 0, 0. The
// caller takes care of all the other details.
//
class ImageReader
{
public:
    virtual xySize
    getData(string const &filename, vector<float> &dataContainer, ColorChannel_t channel = NNet::R) = 0;
};

class ImageReaderBMP : public ImageReader
{
public:
    xySize getData(string const &filename, vector<float> &dataContainer, ColorChannel_t channel) override;
};

class ImageReaderDat : public ImageReader
{
public:
    xySize getData(string const &filename, vector<float> &dataContainer, ColorChannel_t channel) override;
};


// One Sample holds one set of neural net input values, and the expected output
// values (if known in advance).
//
class Sample
{
public:
    // Returns a reference to the cached image data, flattened in a 1D container:
    vector<float> const &getData(ColorChannel_t channel);

    // Clear all cached image data (does not clear data that was explicitly defined):
    void clearImageCache(void);

    string imageFilename; // Ignored for explicit data
    xySize size;  // X, Y image dimensions, nonzero if valid

    // Data caches:
    vector<float> targetVals;
    vector<float> data;
};


// A SampleSet object holds a container of all the input samples to be processed
// by the neural net. It also manages the image file readers.
//
class SampleSet
{
public:
    void loadSamples(string const &inputDataConfigFilename);
    void shuffle(void);          // Shuffles the samples container
    void clearImageCache(void);  // Only image data is cleared, not explicit input data

    static vector<ImageReader *> imageReaders; // One for each supported image format
    vector<Sample> samples;
};


class Neuron;     // Forward references
class Connection;

typedef vector<float> matColumn_t;
typedef vector<matColumn_t> mat2D_t; // Allows access as mat[x][y]

typedef float (*transferFunction_t)(float); // Also used for the derivative function


// This structure holds the information extracted from a single line in
// the topology config file. The topology file parser creates one of these
// objects for each line in the config file. The parser fills in as much
// of the data as it can. Net::parseConfigFile() is responsible for
// converting a container of topologyConfigSpec_t into layers of neurons.
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

    string layerName;                  // Can be input, output, or layer*
    bool isRegularLayer;
    bool isConvolutionFilterLayer;     // Equivalent to (convolveMatrix.size() == 1)
    bool isConvolutionNetworkLayer;    // Equivalent to (convolveMatrix.size() > 1)
    bool isPoolingLayer;               // Equivalent to (poolSize.x != 0)
    dxySize size;                      // layer depth, X, Y dimensions
    ColorChannel_t channel;            // Applies only to the input layer
    xySize radius;                     // Always used, so set high to fully connect layers
    string transferFunctionName;

    // The rest of the members below are used by convolution and pooling layers only:

    enum poolMethod_t poolMethod;      // Used only for pooling layers
    xySize poolSize;

    // In the flatConvolveMatrix container, the size of the outer container equals
    // the layer depth, and the inner container contains the convolution kernel
    // flattened into a 1D array:

    vector<vector<float>> flatConvolveMatrix;  // Inner index = x*szY + y
    xySize kernelSize;                 // Used only for convolution layers
};


//  ***********************************  class Layer  ***********************************

// Each layer conceptually manages a bag of neurons in a 2D arrangement, stored
// flattened in a 1D container:
//
class Layer
{
public: // New
    Layer(const topologyConfigSpec_t &params);
    vector<vector<Neuron>> neurons;    // neurons[depth][i], where i = flattened 2D index
    string layerName;                  // Can be input, output, or layer*
    dxySize size;                      // layer depth, X, Y dimensions (number of neurons)
    bool isRegularLayer;
    bool isConvolutionFilterLayer;     // Equivalent to (convolveMatrix.size() == 1)
    bool isConvolutionNetworkLayer;    // Equivalent to (convolveMatrix.size() > 1)
    bool isPoolingLayer;               // Equivalent to (poolSize.x != 0)
    ColorChannel_t channel;            // Applies only to the input layer
    xySize radius;                     // Always used in regular layers, so set high to fully connect layers
    transferFunction_t tf;             // Ignored by convolution filter layers
    transferFunction_t tfDerivative;   // Ignored by convolution filter layers
    vector<Connection> *pConnections;  // Pointer to the container of all Connection records
    uint32_t totalNumberBackConnections;
    bool projectRectangular = false;   // Defines shape when radius parameter is used

    // In these containers, the size of the outer container equals the layer depth,
    // and the inner container contains the convolution kernel flattened into a 1D array:
    vector<vector<float>> flatConvolveMatrix;  // Inner index = x*szY + y
    vector<vector<float>> flatConvolveGradients;
    vector<vector<float>> flatDeltaWeights;
    xySize kernelSize;                 // Used only for convolution layers

    enum poolMethod_t poolMethod;      // Used only for pooling layers
    xySize poolSize;                   // Used only for pooling layers

    static void clipToBounds(int32_t &xmin, int32_t &xmax, int32_t &ymin, int32_t &ymax, dxySize &size);
    virtual void saveWeights(std::ofstream &);
    virtual void loadWeights(std::ifstream &);
    void connectLayers(Layer &layerFrom);
    void connectOneNeuronAllDepths(Layer &fromLayer, Neuron &toNeuron,
                uint32_t destDepth, uint32_t destX, uint32_t destY);
    void connectBiasToAllNeuronsAllDepths(Neuron &bias);
    void resolveTransferFunctionName(string const &transferFunctionName);
    virtual void debugShow(bool details);
    virtual void calcGradients(const vector<float> &targetVals);
    virtual void updateWeights(float eta, float alpha);
    virtual void feedForward() = 0;

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    virtual std::string visualizationsAvailable(void); // Creates options for the drop-down menu in the GUI
    virtual string visualizeKernels(void); // Create a base64-encoded BMP image
    virtual string visualizeOutputs(void); // Create a base64-encoded BMP image
#endif
};

class LayerRegular : public Layer
{
public:
    LayerRegular(const topologyConfigSpec_t &params);
    void feedForward();
    void saveWeights(std::ofstream &);
    void loadWeights(std::ifstream &);
    void updateWeights(float eta, float alpha);
    void debugShow(bool details);
#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    string visualizationsAvailable(void);
#endif
};

class LayerConvolution : public Layer
{
public:
    LayerConvolution(const topologyConfigSpec_t &params);
    void feedForward();
#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    string visualizationsAvailable(void);
    string visualizeKernels(void);
#endif
};

class LayerConvolutionFilter : public LayerConvolution
{
public:
    LayerConvolutionFilter(const topologyConfigSpec_t &params);
    void debugShow(bool details);
};

class LayerConvolutionNetwork : public LayerConvolution
{
public:
    LayerConvolutionNetwork(const topologyConfigSpec_t &params);
    void calcGradients(const vector<float> &targetVals);
    void updateWeights(float eta, float alpha);
    void saveWeights(std::ofstream &);
    void loadWeights(std::ifstream &);
    void debugShow(bool details);
};

class LayerPooling : public Layer
{
public:
    LayerPooling(const topologyConfigSpec_t &params);
    void feedForward();
    void updateWeights(float eta, float alpha);
    void debugShow(bool details);
#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    string visualizationsAvailable(void);
#endif
};


// ***********************************  class Connection  ***********************************


// If neurons are considered as nodes in a directed graph, the edges would be the
// "connections". Each connection goes from one neuron to any other neuron in any
// layer. Connections are analogous to synapses in natural biology. The .weight
// member is what it's all about -- once a net is trained, we only need to save
// the weights of all the connections. The set of all weights plus the network
// topology defines the neural net's function. The .deltaWeight member is used
// only for the momentum calculation. For convolution layers, the weights are
// stored in the Layer class, not here.
//
class Connection
{
public:
    Connection(Neuron &from, Neuron &to);
    Neuron &fromNeuron;
    Neuron &toNeuron;
    float weight;       // Used only by regular neuron layers
    float deltaWeight;  // The weight change from the previous training iteration

    // Used only by convolution layers, this is an index into the flattened convolve
    // matrix container and the associated gradient container:
    int convolveMatrixIndex;
};


// ***********************************  class Neuron  ***********************************


class Neuron
{
public:
    Neuron();
    float output;
    float gradient;

    // All the input and output connections for this neuron. We store these as indices
    // into an array of Connection objects stored somewhere else. We store indices
    // instead of pointers or iterators because the container of Connection objects
    // can get reallocated which would invalidate the references. Currently the connections
    // container is a public data member of class Net, but it could be stored anywhere that
    // is accessible.

    vector<uint32_t> backConnectionsIndices;    // My back connections
    vector<uint32_t> forwardConnectionsIndices; // My forward connections

    void feedForward(Layer *pMyLayer);          // Propagate the net inputs to the outputs
    void feedForwardConvolution(uint32_t depth, Layer *pMyLayer); // Special for conv. network layers
    void feedForwardPooling(Layer *pMyLayer);   // Special for pooling layers
    void updateInputWeights(float eta, float alpha, vector<Connection> *pConnections); // For backprop training
    void updateInputWeightsConvolution(uint32_t depth, float eta, float alpha, Layer &myLayer);
    void calcOutputGradients(float targetVal, transferFunction_t tfDerivative); // For backprop training
    void calcHiddenGradients(Layer &myLayer);   // For backprop training
    void calcHiddenGradientsConvolution(uint32_t depth, Layer &myLayer); // Special for convolution network layers

    // The only reason for the .sourceNeurons member is to make it easy to
    // find and report any unconnected neurons. For everything else, we'll use
    // the backConnectionIndices to find the neuron's inputs. This container
    // holds pointers to all the source neurons that feed this neuron:
    std::set<Neuron *> sourceNeurons;

private:
    float sumDOW_nextLayer(vector<Connection> *pConnections) const; // Used in hidden layer backprop training
};


// ***********************************  class Net  ***********************************


class Net
{
public: // This public section exposes the complete public API for class Net

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

    // The second ctor parameter is used by the unit tests to disable the webserver even
    // if it was compiled in. You can use the preprocessor macro -DDISABLE_WEBSERVER to
    // prevent the webserver code from being compiled and linked.
    //
    Net(const string &topologyFilename, bool webserverEnabled = true); // ctor
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

    SampleSet sampleSet;     // Manages all things about input data

public: // These members are public only for convenience of unit testing:
    uint32_t inputSampleNumber; // Increments each time feedForward() is called

    static const uint32_t HUGE_RADIUS = (uint32_t)1e9; // Magic value

    // Here is where we store all the weighted connections. The container can get
    // reallocated, so we'll only refer to elements by indices, not by pointers or
    // references.
    vector<Connection> connections;

    vector<std::unique_ptr<Layer>> layers; // Polymorphic

    Neuron bias;  // Fake neuron with constant output 1.0
    float lastRecentAverageError;    // Used for dynamically adjusting eta
    uint32_t totalNumberBackConnections; // Including 1 bias connection per neuron
    uint32_t totalNumberNeurons;
    vector<topologyConfigSpec_t> parseTopologyConfig(std::istream &cfg);
    void configureNetwork(vector<topologyConfigSpec_t> configSpecs, const string configFilename = "");
    void reportUnconnectedNeurons(void);
    float adjustedEta(void);

private:
    void parseConfigFile(const string &configFilename); // Creates layer metadata from a config file
    Layer &createLayer(const topologyConfigSpec_t &params);
    bool addConnectionsToLayer(Layer &layerTo, Layer &layerFrom);
    void createAllNeurons(Layer &layerTo, Layer &layerFrom);
    int32_t getLayerNumberFromName(string &name) const;

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    bool webserverEnabled;    // false to disable at runtime
    void doCommand(); // Handles incoming program command and control
    void actOnMessageReceived(Message_t &msg);
    void makeParameterBlock(string &s);
    string visualizationMenu; // Options for the drop-down menu in the GUI
    Layer *pLayerToVisualize; // Only one layer can be visualized at a time; NULL to disable visualizations
    string visualizeChoice;   // Layer-dependent, what visualization to do
    WebServer webServer;
    int portNumber;
    MessageQueue messages;
#endif
};

extern uint32_t flattenXY(uint32_t x, uint32_t y, uint32_t ySize);
extern uint32_t flattenXY(uint32_t x, uint32_t y, dxySize size);

} // end namespace NNet

#endif // end #ifndef NNET_H
