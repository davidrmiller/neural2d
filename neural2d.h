/*
neural2d.h
David R. Miller, 2014
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
 *   11. Tutorial video coming soon!
 *
 * This program is written in the C++-11 dialect. It uses mostly ISO-standard C++
 * features and a few POSIX features that should be widely available on any
 * compiler that knows about C++11 and POSIX.
 *
 * This is a console program that requires no GUI. However, a GUI is optional:
 * The WebServer class provides an HTTP server that a web browser can connect to
 * at port 24080.
 *
 * For most array indices and offsets into containers, we use 32-bit integers
 * (uint32_t). This allows 2 billion neurons per layer, 2 billion connections
 * total, etc.
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

#include <unistd.h>    // For sleep()


using namespace std;


// Everything we define in this file will be inside the NNet namespace. This keeps
// all of our definitions out of the global namespace. (You can indent all the source
// lines inside the namespace below if you're an indenting purist.)
//
namespace NNet {


enum ColorChannel_t { R, G, B, BW };


class Sample
{
public:
    string imageFilename;
    vector<float> &getData(ColorChannel_t colorChannel); // Review !!!
    void clearCache(void);
    vector<float> targetVals;
    vector<float> data; // Pixel data converted to floats and flattened to a 1D array
};


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


class Neuron; // Forward reference

typedef vector<float> matColumn_t;
typedef vector<matColumn_t> convolveMatrix_t; // Allows access as convolveMatrix[x][y]

typedef float (*transferFunction_t)(float); // Also used for the derivative function

struct layerParams_t {
    layerParams_t(void);
    void resolveTransferFunctionName(void);
    void clear(void);
    string layerName;                  // Can be input, output, or layer*
    string fromLayerName;              // Can be any existing layer name
    uint32_t sizeX;                    // Format: size XxY
    uint32_t sizeY;
    ColorChannel_t channel;            // Applies only to the input layer
    bool colorChannelSpecified;
    uint32_t radiusX;                  // Format: radius XxY
    uint32_t radiusY;
    string transferFunctionName;       // Format: tf name
    transferFunction_t tf;
    transferFunction_t tfDerivative;
    convolveMatrix_t convolveMatrix;   // Format: convolve {{0,1,0},...
    bool isConvolutionLayer;           // Equivalent to (convolveMatrix.size() != 0)
};


//  ***********************************  class Layer  ***********************************

// Each layer is a bag of neurons in a 2D arrangement:
//
struct Layer
{
    layerParams_t params;

    // For each layer, before any references are made to its members, the .neurons
    // member must be initialized with sufficient capacity to prevent reallocation.
    // This allows us to form stable pointers, iterators, or references to neurons.
    vector<Neuron> neurons;  // 2D array, flattened index = y * sizeX + x
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
    Neuron(vector<Connection> *pConnectionsData, transferFunction_t tf, transferFunction_t tfDerivative);
    float output;
    float gradient;

    // All the input and output connections for this neuron. We store these as indices
    // into an array of Connection objects stored somewhere else. We store indices
    // instead of pointers or iterators because the container of Connection objects
    // can get reallocated which would invalidate the references. Each neuron gets
    // a pointer to the connections container. Currently the connections container
    // is a public data member of class Net, but it could be stored anywhere that is
    // accessible.

    vector<Connection> *pConnections;   // Pointer to the container of Connection records
    vector<uint32_t> backConnectionsIndices;
    vector<uint32_t> forwardConnectionsIndices; // Refers to another neuron's back connections

    void feedForward(void);                     // Propagate the net inputs to the outputs
    void updateInputWeights(float eta, float alpha);  // For backprop training
    void calcOutputGradients(float targetVal);         // For backprop training
    void calcHiddenGradients(void);                     // For backprop training

    // The only reason for the .sourceNeurons member is to make it easy to
    // find and report any unconnected neurons. For everything else, we'll use
    // the backConnectionIndices to find the neuron's inputs. This container
    // holds pointers to all the source neurons that feed this neuron:

    set<Neuron *> sourceNeurons;

private:
    float (*transferFunction)(float x);
    float (*transferFunctionDerivative)(float x);
    float sumDOW_nextLayer(void) const;        // Used in hidden layer backprop training
};


// ***********************************  class Net  ***********************************

class Net
{
public:
    // Parameters that affect overall network operation. These can be set by
    // directly accessing the data members:

    bool enableBackPropTraining;   // If false, backProp() won't update any weights

    // Training will pause when the recent average overall error falls below this threshold:

    float doneErrorThreshold;

    // eta is the network learning rate. It can be set to a constant value, somewhere
    // in the range 0.0001 to 0.1. Optionally, set dynamicEtaAdjust to true to allow
    // the program to automatically adjust eta during learning for optimal learning.

    float eta;                // Initial overall net learning rate, [0.0..1.0]
    bool dynamicEtaAdjust;    // true enables automatic eta adjustment during training

    // alpha is the momentum factor. Set it to zero to disable momentum. If alpha > 0, then
    // changes in any connection weight in the same direction that the weight was changed
    // last time is amplified. This helps converge on a solution a little faster during
    // the early stages of training, but if set too high will interfere with the network
    // converging on the most accurate solution.

    float alpha;              // Initial momentum, multiplier of last deltaWeight, [0.0..1.0]

    // Regularization parameter. If zero, regularization is disabled:

    float lambda;

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
    // If shuffleInputSamplies is true, then the input samples will be randomly
    // shuffled after each use:

    bool repeatInputSamples;
    bool shuffleInputSamples;

    string weightsFilename;     // Filename to use in saveWeights() and loadWeights()

    uint32_t inputSampleNumber; // Increments each time feedForward() is called
    float error;                // Overall net error
    float recentAverageError;   // Averaged over recentAverageSmoothingFactor samples

    // Creates and connects a net from a topology config file:

    Net(const string &topologyFilename);
    ~Net(void);

    void feedForward(void);                       // Propagate inputs to outputs
    void feedForward(Sample &sample);
    void backProp(const Sample &sample);          // Backprop and update all weights

    // The connection weights can be saved or restored at any time. Note that the network
    // topology is not saved in the weights file, so you'll have to manually keep track of
    // which weights file goes with which topology file.

    bool saveWeights(const string &filename) const;
    bool loadWeights(const string &filename);

    // Functions for forward propagation:

    float getNetError(void) const { return error; };
    float getRecentAverageError(void) const { return recentAverageError; };
    void calculateOverallNetError(const Sample &sample);  // Update .error member

    // Functions for displaying the results when processing input samples:

    void reportResults(const Sample &sample) const;
    void debugShowNet(bool details = false);      // Display details of net topology

    SampleSet sampleSet;     // List of input images and access to their data

private:
    // Here is where we store all the weighted connections. The container can get
    // reallocated, so we'll only refer to elements by indices, not by pointers or
    // references. This also allows us to hack on the connections container during
    // training if we want to, by dynamically adding or deleting connections without
    // invalidating references.

    vector<Connection> connections;

    Neuron bias;  // Fake neuron with constant output 1.0
    void initTrainingSamples(const string &inputFilename);
    void parseConfigFile(const string &configFilename); // Creates layer metadata from a config file
    convolveMatrix_t parseMatrixSpec(istringstream &ss);
    Layer &createLayer(const layerParams_t &params);
    bool addToLayer(Layer &layerTo, Layer &layerFrom, layerParams_t &params);
    void createNeurons(Layer &layerTo, Layer &layerFrom, layerParams_t &params);
    void connectNeuron(Layer &layerTo, Layer &layerFrom, Neuron &neuron,
                       uint32_t nx, uint32_t ny, layerParams_t &params);
    void connectBias(Neuron &neuron);
    int32_t getLayerNumberFromName(const string &name) const;
    float adjustedEta(void);

    vector<Layer> layers;

    float lastRecentAverageError;    // Used for dynamically adjusting eta
    uint32_t totalNumberConnections; // Including 1 bias connection per neuron
    uint32_t totalNumberNeurons;

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    void doCommand(); // Handles incoming program command and control
    void actOnMessageReceived(Message_t &msg);
    void makeParameterBlock(string &s);
    WebServer webServer;
    int portNumber;
    MessageQueue messages;
#endif
};

} // end namespace NNet

#endif // end #ifndef NNET_H
