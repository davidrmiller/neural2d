/*
neural2d-core.cpp
David R. Miller, 2014, 2015
https://github.com/davidrmiller/neural2d

See neural2d.h for more information.
*/

#include <cctype>
#include <cmath>
#include <memory>   // for unique_ptr
#include <unistd.h> // For sleep() or usleep()

#include "neural2d.h"

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
#include "webserver.h"
#endif


namespace NNet {


//  ***********************************  Utility functions  ***********************************


// Logger objects: these can be used like cout and cerr:
//
Logger info, warn, err(std::cerr);


// Returns a random float in the range [0.0..1.0]
//
float randomFloat(void)
{
    return (float)rand() / RAND_MAX;
}


// Given a (depth, x, y) coordinate, return a flattened index.
// There's nothing magic here; we use a function to do this so that we
// always flatten it the same way each time:
//
uint32_t flattenDXY(uint32_t depth, uint32_t x, uint32_t y, uint32_t xSize, uint32_t ySize)
{
    return depth * (xSize*ySize) + y * xSize + x;
}

uint32_t flattenDXY(uint32_t depth, uint32_t x, uint32_t y, dxySize size)
{
    return flattenDXY(depth, x, y, size.x, size.y);
}

uint32_t flattenDXY(dxySize coord3d, dxySize size)
{
    return flattenDXY(coord3d.depth, coord3d.x, coord3d.y, size.x, size.y);
}


// Assuming an ellipse centered at 0,0 and aligned with the global axes, returns
// a positive value if x,y is outside the ellipse; 0.0 if on the ellipse;
// negative if inside the ellipse.
//
float elliptDist(float x, float y, float radiusX, float radiusY)
{
    assert(radiusX >= 0.0 && radiusY >= 0.0);
    return radiusY*radiusY*x*x + radiusX*radiusX*y*y - radiusX*radiusX*radiusY*radiusY;
}


bool isFileExists(string const &filename)
{
    std::ifstream file(filename);
    return (bool)file;
}


// Add overloads as needed:
int32_t max(int32_t a, int32_t b) { return a >= b ? a : b; }
int32_t min(int32_t a, int32_t b) { return a <= b ? a : b; }
float absd(float a) { return a < 0.0 ? -a : a; }


// Replaces potentially dangerous chars with underscores
//
void sanitizeFilename(string &s)
{
    for (char &c : s) {
        if (!isalnum(c) && c != '_' && c != '-' && c != '%' && c!= '.') {
            c = '_';
        }
    }
}


// ***********************************  Transfer Functions  ***********************************

// Here is where we define at least one transfer function. We refer to them by
// name, where "" is an alias for the default function. To select a different one,
// add a "tf" parameter to the layer definition in the topology config file. All the
// neurons in any one layer will use the same transfer function.

// tanh is a sigmoid curve scaled; output ranges from -1 to +1:
float transferFunctionTanh(float x) { return tanh(x); }
float transferFunctionDerivativeTanh(float x) { return 1.0 - tanh(x) * tanh(x); }

// logistic is a sigmoid curve that ranges 0.0 to 1.0:
float transferFunctionLogistic(float x) { return 1.0 / (1.0 + exp(-x)); }
float transferFunctionDerivativeLogistic(float x) { return exp(-x) / pow((exp(-x) + 1.0), 2.0); }

// linear is a constant slope; ranges from -inf to +inf:
float transferFunctionLinear(float x) { return x; }
float transferFunctionDerivativeLinear(float x) { return (void)x, 1.0; }

// ramp is a constant slope between -1 <= x <= 1, zero slope elsewhere; output ranges from -1 to +1:
float transferFunctionRamp(float x)
{
    if (x < -1.0) return -1.0;
    else if (x > 1.0) return 1.0;
    else return x;
}
float transferFunctionDerivativeRamp(float x) { return (x < -1.0 || x > 1.0) ? 0.0 : 1.0; }

// gaussian:
float transferFunctionGaussian(float x) { return exp(-((x * x) / 2.0)); }
float transferFunctionDerivativeGaussian(float x) { return -x * exp(-(x * x) / 2.0); }

float transferFunctionIdentity(float x) { return x; } // Used only in convolution layers
float transferFunctionIdentityDerivative(float x) { return (void)x, 1.0; }


void layerParams_t::resolveTransferFunctionName(void)
{
    if (transferFunctionName == "" || transferFunctionName == "tanh") {
        // This is the default transfer function:
        tf = transferFunctionTanh;
        tfDerivative = transferFunctionDerivativeTanh;
    } else if (transferFunctionName == "logistic") {
        tf = transferFunctionLogistic;
        tfDerivative = transferFunctionDerivativeLogistic;
    } else if (transferFunctionName == "linear") {
        tf = transferFunctionLinear;
        tfDerivative = transferFunctionDerivativeLinear;
    } else if (transferFunctionName == "ramp") {
        tf = transferFunctionRamp;
        tfDerivative = transferFunctionDerivativeRamp;
    } else if (transferFunctionName == "gaussian") {
        tf = transferFunctionGaussian;
        tfDerivative = transferFunctionDerivativeGaussian;
    } else if (transferFunctionName == "identity") {
        tf = transferFunctionIdentity;
        tfDerivative = transferFunctionIdentityDerivative;
    } else {
        err << "Undefined transfer function: \'" << transferFunctionName << "\'" << endl;
        throw exceptionConfigFile();
    }
}


layerParams_t::layerParams_t(void)
{
    clear();
}

void layerParams_t::clear(void)
{
    size.depth = size.x = size.y = 0;
    channel = NNet::BW;
    radius.x = Net::HUGE_RADIUS;
    radius.y = Net::HUGE_RADIUS;
    transferFunctionName.clear();
    transferFunctionName = "tanh";
    tf = transferFunctionTanh;
    tfDerivative = transferFunctionDerivativeTanh;
    convolveMatrix.clear();
    isConvolutionFilterLayer = false;   // Equivalent to (convolveMatrix.size() == 1)
    isConvolutionNetworkLayer = false;  // Equivalent to (convolveMatrix.size() > 1)
    kernelSize.x = kernelSize.y = 0;    // Used only for convolution network layers
    poolSize.x = poolSize.y = 0;        // Used only for convolution network layers
    poolMethod = POOL_NONE;
}


// ***********************************  Input samples  ***********************************

// Given an image filename and a data container, fill the container with
// data extracted from the image, using the conversion function specified
// in colorChannel:
//
void ReadBMP(const string &filename, vector<float> &dataContainer, ColorChannel_t colorChannel)
{
    FILE* f = fopen(filename.c_str(), "rb");

    if (f == NULL) {
        err << "Error reading image file \'" << filename << "\'" << endl;
        // To do: add appropriate error recovery here
        throw exceptionImageFile();
    }

    // Read the BMP header to get the image dimensions:

    unsigned char info[54];
    if (fread(info, sizeof(unsigned char), 54, f) != 54) {
        err << "Error reading the image header from \'" << filename << "\'" << endl;
        throw exceptionImageFile();
    }

    if (info[0] != 'B' || info[1] != 'M') {
        err << "Error: invalid BMP file \'" << filename << "\'" << endl;
        throw exceptionImageFile();
    }

    // Verify the offset to the pixel data. It should be the same size as the info[] data read above.

    size_t dataOffset = (info[13] << 24)
                      + (info[12] << 16)
                      + (info[11] << 8)
                      +  info[10];

    // Verify that the file contains 24 bits (3 bytes) per pixel (red, green blue at 8 bits each):

    int pixelDepth = (info[29] << 8) + info[28];
    if (pixelDepth != 24) {
        err << "Error: BMP file is not 24 bits per pixel" << endl;
        throw exceptionImageFile();
    }

    // This method of converting 4 bytes to a uint32_t is portable for little- or
    // big-endian environments:

    uint32_t width = (info[21] << 24)
                   + (info[20] << 16)
                   + (info[19] << 8)
                   +  info[18];

    uint32_t height = (info[25] << 24)
                    + (info[24] << 16)
                    + (info[23] << 8)
                    +  info[22];

    // Position the read pointer to the first byte of pixel data:

    if (fseek(f, dataOffset, SEEK_SET) != 0) {
        err << "Error seeking in BMP file" << endl;
        throw exceptionImageFile();
    }

    uint32_t rowLen_padded = (width*3 + 3) & (~3);
    std::unique_ptr<unsigned char[]> imageData {new unsigned char[rowLen_padded]};

    dataContainer.clear();

    // Fill the data container with 8-bit data taken from the image data:

    for (uint32_t y = 0; y < height; ++y) {
        if (fread(imageData.get(), sizeof(unsigned char), rowLen_padded, f) != rowLen_padded) {
            err << "Error reading \'" << filename << "\' row " << y << endl;
            // To do: add appropriate error recovery here
            throw exceptionImageFile();
        }

        // BMP pixels are arranged in memory in the order (B, G, R). We'll convert
        // the pixel to a float using one of the conversions below:

        float val = 0.0;

        for (uint32_t x = 0; x < width; ++x) {
            if (colorChannel == NNet::R) {
                val = imageData[x * 3 + 2]; // Red
            } else if (colorChannel == NNet::G) {
                val = imageData[x * 3 + 1]; // Green
            } else if (colorChannel == NNet::B) {
                val = imageData[x * 3 + 0]; // Blue
            } else if (colorChannel == NNet::BW) {
                val =  0.3 * imageData[x*3 + 2] +   // Red
                       0.6 * imageData[x*3 + 1] +   // Green
                       0.1 * imageData[x*3 + 0];    // Blue
            } else {
                err << "Error: unknown pixel conversion" << endl;
                throw exceptionImageFile();
            }

            // Convert the pixel value to the range -1.0..1.0:
            // This value will be the input to an input neuron:
            dataContainer.push_back(val / 128.0 - 1.0);

            // Alternatively:
            // Convert the pixel value to the range 0.0..1.0:
            //dataContainer.push_back(val / 256.0);
        }
    }

    fclose(f);
}


// If the data is available, we'll return it. If this is the first time getData()
// is called for inputs that come from an image, we'll open the image file and
// cache the pixel data in memory. Returns a reference to the container of input data.
//
vector<float> &Sample::getData(ColorChannel_t colorChannel)
{
   if (data.size() == 0 && imageFilename != "") {
       ReadBMP(imageFilename, data, colorChannel);
   }

   // If we get here, we can assume there is something in the .data member
   // To do: check that

   return data;
}


void Sample::clearCache(void)
{
    data.clear();
}


// Given the name of an input sample config file, open it and save the contents
// in memory. For lines that specify an image filename, we'll just save the filename
// for now and defer reading the pixel data until it's needed. For lines that
// contain an explicit list of input values, we'll save the values. The syntax
// for a line specifying a filename is of the form:
//     filename t1 t2 t3...
// where t1, t2, etc. are the target output values. The syntax for a line that
// specifies explicit values is:
//     { i1, i2, i3... } t1 t2 t3
// where i1, i2... are the input values and t1, t2, etc. are the target output values.
//
void SampleSet::loadSamples(const string &inputFilename)
{
    string line;
    uint32_t lineNum = 0;

    if (!isFileExists(inputFilename)) {
        err << "Error reading input samples config file \'" << inputFilename << "\'" << endl;
        throw exceptionInputSamplesFile();
    }

    std::ifstream dataIn(inputFilename);
    if (!dataIn || !dataIn.is_open()) {
        err << "Error opening input samples config file \'" << inputFilename << "\'" << endl;
        throw exceptionInputSamplesFile();
    }

    samples.clear();  // Lose all prior samples

    while (getline(dataIn, line)) {
        ++lineNum;
        Sample sample; // Default ctor will clear all members
        string token;
        char delim;

        std::stringstream ss(line);
        ss >> token;
        if (token == "{") {
            // This means we have literal values like "{ 0.2 0 -1.0}"
            sample.imageFilename="";   // "" means we have immediate data

            // Read from the char after the { up to but including the } char:
            char args[16384]; // Review !!!
            ss.get(args, sizeof args, '}');
            std::stringstream inargs(args);
            while (!inargs.eof()) {
                float val;
                if (!(inargs >> val).fail()) {
                    sample.data.push_back(val);
                }
            }
            ss >> delim;
        } else {
            // We may have an image filename (instead of an explicit list of values):
            sample.imageFilename = token;
            // Skip blank and comment lines:
            if (sample.imageFilename.size() == 0 || sample.imageFilename[0] == '#') {
                continue;
            }
        }

        // If they exist, read the target values from the rest of the line:
        while (!ss.eof()) {
            float val;
            if (!(ss >> val).fail()) {
                sample.targetVals.push_back(val);
            }
        }

        samples.push_back(sample);
    }

    info << samples.size() << " training samples initialized" << endl;
}


// Randomize the order of the samples container.
//
void SampleSet::shuffle(void)
{
    std::random_shuffle(samples.begin(), samples.end());
}


// By clearing the cache, future image access will cause the pixel data to
// be re-read and converted by whatever color conversion is in effect then.
//
void SampleSet::clearCache(void)
{
    for (auto &samp : samples) {
        samp.clearCache();
    }
}

void SampleSet::clearImageCache(void)
{
    for (auto &samp : samples) {
        if (samp.imageFilename != "")
            samp.clearCache();
    }
}


// ***********************************  struct Connection  ***********************************


Connection::Connection(Neuron &from, Neuron &to)
     : fromNeuron(from), toNeuron(to)
{
    deltaWeight = 0.0;
}


// ***********************************  class Neuron  ***********************************


// This default ctor is used only when defining the special bias neuron in the
// Net class. The bias neuron only feeds forward a constant value of 1.0,
// so it doesn't use a transfer function, and it doesn't need access to the
// connections container because there are no back connections.
// Alternatively, the bias neuron could have been defined as a subclass of
// class Neuron.
//
Neuron::Neuron()
{
    output = 1.0;
    gradient = 0.0;
    backConnectionsIndices.clear();
    forwardConnectionsIndices.clear();
    sourceNeurons.clear();
}


// All neurons (except the special bias neuron in the Net class) use this ctor
// to construct new neurons. The transfer function is a required ctor parameter
// because once a transfer function is selected, it shouldn't be changed (if it
// were to change, all the weights would suddenly be wrong).
//
Neuron::Neuron(const Layer *pMyLayer_)
    : pMyLayer(pMyLayer_)
{
    output = randomFloat() - 0.5;
    gradient = 0.0;
    backConnectionsIndices.clear();
    forwardConnectionsIndices.clear();
    sourceNeurons.clear();
}


// The error gradient of an output-layer neuron is equal to the target (desired)
// value minus the computed output value, times the derivative of
// the output-layer activation function evaluated at the computed output value.
//
void Neuron::calcOutputGradients(float targetVal)
{
    float delta = targetVal - output;
    gradient = delta * pMyLayer->params.tfDerivative(output);
}


// The error gradient of a hidden-layer neuron is equal to the derivative
// of the activation function of the hidden layer evaluated at the
// local output of the neuron times the sum of the product of
// the primary outputs times their associated hidden-to-output weights.
//
void Neuron::calcHiddenGradients(void)
{
    float dow = sumDOW_nextLayer();
    gradient = dow * pMyLayer->params.tfDerivative(output);
}


// To do: add commentary!!!
//
float Neuron::sumDOW_nextLayer(void) const
{
    float sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (auto idx : forwardConnectionsIndices) {
        const Connection &conn = (*pMyLayer->pConnections)[idx];

        sum += conn.weight * conn.toNeuron.gradient;
    }

    return sum;
}


void Neuron::updateInputWeights(float eta, float alpha)
{
    // The weights to be updated are the weights from the neurons in the
    // preceding layer (the source layer) to this neuron:

    // Optionally enable the following #pragma line to permit OpenMP to
    // parallelize this loop. For clang or gcc, add the option "-fopenmp" to
    // the compiler command line. If the compiler does not understand OpenMP,
    // the #pragma will be ignored. This particular parallelization does not
    // gain much performance, perhaps because the memory accesses during
    // backprop are so randomly distributed, thwarting the cache mechanisms.

//#pragma omp parallel for
    for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
        auto idx = backConnectionsIndices[i];
        Connection &conn = (*pMyLayer->pConnections)[idx];

        const Neuron &fromNeuron = conn.fromNeuron;
        float oldDeltaWeight = conn.deltaWeight;

        float newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * fromNeuron.output
                * gradient
                // Add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        conn.deltaWeight = newDeltaWeight;
        conn.weight += newDeltaWeight;
    }
}


// To feed forward an individual neuron, we'll sum the weighted inputs, then pass that
// sum through the transfer function.
//
void Neuron::feedForward(void)
{
    float sum = 0.0;

    // Sum the neuron's inputs:
    for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
        size_t idx = backConnectionsIndices[i];
        const Connection &conn = (*pMyLayer->pConnections)[idx];

        sum += conn.fromNeuron.output * conn.weight;
    }

    // Shape the output by passing it through the transfer function:
    this->output = pMyLayer->params.tf(sum);
}


// ***********************************  class Net  ***********************************


Net::Net(const string &topologyFilename)
{
    // See nnet.h for descriptions of these variables:

    enableBackPropTraining = true;
    doneErrorThreshold = 0.001;
    eta = 0.01;                    // Initial overall net learning rate, [0.0..1.0]
    dynamicEtaAdjust = true;       // true enables automatic eta adjustment during training
    alpha = 0.1;                   // Momentum factor, multiplier of last deltaWeight, [0.0..1.0]
    lambda = 0.0;                  // Regularization parameter; disabled if 0.0
    projectRectangular = false;    // Use elliptical areas for sparse connections
    isRunning = true;              // Command line option -p overrides this
    reportEveryNth = 1;
    recentAverageSmoothingFactor = 125.; // Average net errors over this many input samples
    repeatInputSamples = true;
    shuffleInputSamples = true;
    weightsFilename = "weights.txt";
    inputSampleNumber = 0;         // Increments each time feedForward() is called
    error = 1.0;
    recentAverageError = 1.0;
    connections.clear();
    layers.clear();
    lastRecentAverageError = 1.0;
    totalNumberConnections = 0;
    totalNumberNeurons = 0;

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    portNumber = 24080;
    webServer.start(portNumber, messages);
#endif

    // Initialize the dummy bias neuron to provide a weighted bias input for all other neurons.
    // This is a single special neuron that has no inputs of its own, and feeds a constant
    // 1.0 through weighted connections to every other neuron in the network except input
    // neurons:

    bias.output = 1.0;
    bias.gradient = 0.0;

    // Set up the layers, create neurons, and connect them:

    if (topologyFilename.size() > 0) {
        parseConfigFile(topologyFilename);  // Throws an exception if any error
    }
}


Net::~Net(void) {
#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    webServer.stopServer();
#endif
}


// Load weights from an external file. The file must contain one floating point
// number per line, with no blank lines. This function is intended to read the
// same format that saveWeights() produces.
//
bool Net::loadWeights(const string &filename)
{
    if (!isFileExists(filename)) {
        err << "Error reading weights file \'" << filename << "\'" << endl;
        throw exceptionWeightsFile();
    }

    std::ifstream file(filename);
    if (!file) {
        err << "Error reading weights file \'" << filename << "\'" << endl;
        throw exceptionWeightsFile();
    }

    for (auto const &layer : layers) {
        for (auto const &neuron : layer.neurons) {
            for (auto idx : neuron.backConnectionsIndices) {
                Connection &conn = connections[idx];
                file >> conn.weight;
            }
        }
    }

    // ToDo!!! check that the number of weights in the file == size of connections
    file.close();
    return true;
}


// Write all the connection weights to an external file that can be later read
// back in using loadWeights(). The format is one floating point number per
// line, with no blank lines.
//
bool Net::saveWeights(const string &filename) const
{
    std::ofstream file(filename);
    if (!file) {
        err << "Error reading weights file \'" << filename << "\'" << endl;
        throw exceptionWeightsFile();
    }

    for (auto const &layer : layers) {
        for (auto const &neuron : layer.neurons) {
            for (auto idx : neuron.backConnectionsIndices) {
                const Connection &conn = connections[idx];
                file << conn.weight << endl;
            }
        }
    }

    file.close();
    return true;
}


// Assumes the net's output neuron errors and overall net error have already been
// computed and saved in the case where the target output values are known.
//
void Net::reportResults(const Sample &sample) const
{
    // We actually report only every Nth input sample:

    if (inputSampleNumber % reportEveryNth != 0) {
        return;
    }

    // Report actual and expected outputs:

    info << "\nPass #" << inputSampleNumber << ": " << sample.imageFilename << "\nOutputs: ";
    for (auto &n : layers.back().neurons) { // For all neurons in output layer
        info << n.output << " ";
    }
    info << endl;

    if (sample.targetVals.size() > 0) {
        info << "Expected ";
        for (float targetVal : sample.targetVals) {
            info << targetVal << " ";
        }

        // Optional: Enable the following block if you would like to report the net's
        // outputs as a classifier, where the output neuron with the largest output
        // value indicates which class was recognized. This can be used, e.g., for pattern
        // recognition where each output neuron corresponds to one pattern class,
        // and the output neurons are trained to be high to indicate a pattern match,
        // and low to indicate no match.

        if (true) {
            float maxOutput = std::numeric_limits<float>::min();
            size_t maxIdx = 0;

            for (size_t i = 0; i < layers.back().neurons.size(); ++i) {
                Neuron const &n = layers.back().neurons[i];
                if (n.output > maxOutput) {
                    maxOutput = n.output;
                    maxIdx = i;
                }
            }

            if (sample.targetVals[maxIdx] > 0.0) {
                info << " " << string("Correct");
            } else {
                info << " " << string("Wrong");
            }
            info << endl;
        }

        // Optionally enable the following line to display the current eta value
        // (in case we're dynamically adjusting it):
        info << "  eta=" << eta << " ";

        // Show overall net error for this sample and for the last few samples averaged:
        info << "Net error = " << error << ", running average = " << recentAverageError << endl;
    }
}


// Given an existing layer with neurons already connected, add more
// connections. This happens when a layer specification is repeated in
// the config file, thus creating connections to source neurons from
// multiple layers. This applies to regular neurons and neurons in a
// convolution layer. Returns false for any error, true if successful.
//
bool Net::addToLayer(Layer &layerTo, Layer &layerFrom)
{
    for (uint32_t ny = 0; ny < layerTo.params.size.y; ++ny) {
        info << "\r" << ny << std::flush; // Progress indicator

        for (uint32_t nx = 0; nx < layerTo.params.size.x; ++nx) {
            //info << "connect to neuron " << nx << "," << ny << endl;
            connectNeuron(layerTo, layerFrom,
                          layerTo.neurons[flattenDXY(0, nx, ny, layerTo.params.size)],
                          nx, ny);
            // n.b. Bias connections were already made when the neurons were first created.
        }
    }

    info << endl; // End progress indicator

    return true;
}


// Given a layer name and size, create an empty layer. No neurons are created yet.
//
Layer &Net::createLayer(const layerParams_t &params)
{
    layers.push_back(Layer());
    Layer &layer = layers.back(); // Make a convenient name

    layer.params = params;
    layer.params.resolveTransferFunctionName();

    return layer;
}


// This is an optional way to display lots of information about the network
// topology. Tweak as needed. The argument 'details' can be used to control
// if all the connections are displayed in detail.
//
void Net::debugShowNet(bool details)
{
    uint32_t numFwdConnections;
    uint32_t numBackConnections;

    info << "\n\nNet configuration (incl. bias connection): --------------------------" << endl;

    for (auto const &l : layers) {
        numFwdConnections = 0;
        numBackConnections = 0;
        info << "Layer '" << l.params.layerName << "' has " << l.neurons.size()
             << " neurons arranged in " << l.params.size.x << "x" << l.params.size.y << ":" << endl;

        for (auto const &n : l.neurons) {
            if (details) {
                info << "  neuron(" << &n << ")" << " output: " << n.output << endl;
            }

            numFwdConnections += n.forwardConnectionsIndices.size();
            numBackConnections += n.backConnectionsIndices.size(); // Includes the bias connection

            if (details && n.forwardConnectionsIndices.size() > 0) {
                info << "    Fwd connections:" << endl;
                for (auto idx : n.forwardConnectionsIndices) {
                    Connection const &pc = connections[idx];
                    info << "      conn(" << &pc << ") pFrom=" << &pc.fromNeuron
                         << ", pTo=" << &pc.toNeuron
                         << ", w,dw=" << pc.weight << ", " << pc.deltaWeight
                         << endl;
                }
            }

            if (details && n.backConnectionsIndices.size() > 0) {
                info << "    Back connections (incl. bias):" << endl;
                for (auto idx : n.backConnectionsIndices) {
                    Connection const &c = connections[idx];
                    info << "      conn(" << &c << ") pFrom=" << &c.fromNeuron
                         << ", pTo=" << &c.toNeuron
                         << ", w=" << c.weight

                         << endl;
                    assert(&c.toNeuron == &n);
                }
            }
        }

        if (!details) {
            info << "   connections: " << numBackConnections << " back, "
                 << numFwdConnections << " forward." << endl;
        }
    }
}


// Here is where the weights are updated. This is called after every training
// sample. The outputs of the neural net are compared to the target output
// values, and the differences are used to adjust the weights in all the
// connections for all the neurons.
//
void Net::backProp(const Sample &sample)
{
    if (!enableBackPropTraining) {
        return;
    }

    // Calculate output layer gradients:

    Layer &outputLayer = layers.back();
    for (uint32_t n = 0; n < outputLayer.neurons.size(); ++n) {
        outputLayer.neurons[n].calcOutputGradients(sample.targetVals[n]);
    }

    // Calculate hidden layer gradients

    for (uint32_t layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = layers[layerNum]; // Make a convenient name

        for (auto &n : hiddenLayer.neurons) {
            n.calcHiddenGradients();
        }
    }

    // For all layers from outputs to first hidden layer, in reverse order,
    // update connection weights for regular neurons. Skip the update in
    // convolution layers.

    for (uint32_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = layers[layerNum];

        if (!layer.params.isConvolutionFilterLayer) {
            for (auto &neuron : layer.neurons) {
                neuron.updateInputWeights(eta, alpha);
            }
        }
    }

    // Adjust eta if dynamic eta adjustment is enabled:

    if (dynamicEtaAdjust) {
        eta = adjustedEta();
    }
}


// This takes the values at the input layer and feeds them through the
// neural net to produce new values at the output layer.
//
void Net::feedForward(Sample &sample)
{
    ++inputSampleNumber;

    // Move the input data from sample to the input neurons. We'll also
    // check that the number of components of the input sample equals
    // the number of input neurons:

    const vector<float> &data = sample.getData(layers[0].params.channel);
    Layer &inputLayer = layers[0];

    if (inputLayer.neurons.size() != data.size()) {
        err << "Error: input sample " << inputSampleNumber << " has " << data.size()
            << " components, expecting " << inputLayer.neurons.size() << endl;
        //throw exceptionRuntime();
    }

    // Rather than make it a fatal error if the number of input neurons != number
    // of input data values, we'll use whatever we can and skip the rest:

    for (uint32_t i = 0; i < (uint32_t)min(inputLayer.neurons.size(), data.size()); ++i) {
        inputLayer.neurons[i].output = data[i];
    }

    // Start the forward propagation at the first hidden layer:

    for (auto it = layers.begin() + 1; it != layers.end(); ++it){
        for (auto &neuron : it->neurons) {
             neuron.feedForward();
        }
    }

    // If target values are known, update the output neurons' errors and
    // update the overall net error:

    calculateOverallNetError(sample);

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    // Here is a convenient place to poll for incoming commands from the GUI interface:
    doCommand();
#endif
}


// Given the set of target values for the output neurons, calculate
// overall net error (RMS of the output neuron errors). This updates the
// .error and .lastRecentAverageError members. If the container of target
// values is empty, we'll return immediately, leaving the net error == 0.
//
void Net::calculateOverallNetError(const Sample &sample)
{
    error = 0.0;

    // Return if there are no known target values:

    if (sample.targetVals.size() == 0) {
        return;
    }

    const Layer &outputLayer = layers.back();

    // Check that the number of target values equals the number of output neurons:

    if (sample.targetVals.size() != outputLayer.neurons.size()) {
        err << "Error in sample " << inputSampleNumber << ": wrong number of target values" << endl;
        throw exceptionRuntime();
    }

    for (uint32_t n = 0; n < outputLayer.neurons.size(); ++n) {
        float delta = sample.targetVals[n] - outputLayer.neurons[n].output;
        error += delta * delta;
    }

    error /= 2.0 * outputLayer.neurons.size();

    // Regularization calculations -- this is an experimental implementation.
    // If this experiment works, we should instead calculate the sum of weights
    // on the fly during backprop to see if that is better performance.
    // This adds an error term calculated from the sum of squared weights. This encourages
    // the net to find a solution using small weight values, which can be helpful for
    // multiple reasons.

    float sumWeightsSquared_ = 0.0;
    if (lambda != 0.0) {
        for (size_t i = 0; i < connections.size(); ++i) {
            sumWeightsSquared_ += connections[i].weight;
        }

        error += (sumWeightsSquared_ * lambda) / (2.0 * (totalNumberConnections - totalNumberNeurons));
    }

    // Implement a recent average measurement -- average the net errors over N samples:
    lastRecentAverageError = recentAverageError;
    recentAverageError =
            (recentAverageError * recentAverageSmoothingFactor + error)
            / (recentAverageSmoothingFactor + 1.0);
}


// This creates the initial set of connections for a layer of neurons. (If the same layer
// appears again in the topology config file, those additional connections must be added
// to existing connections by calling addToLayer() instead of this function.
//
// Neurons can be "regular" neurons, or convolution filter nodes. If a convolution filter
// matrix is defined for the layer, the neurons in that layer will be connected to source
// neurons in a rectangular pattern defined by the matrix dimensions. No bias connections
// are created for convolution filter nodes. Convolution filter nodes ignore any radius parameter.
// For convolution filter nodes, the transfer function is set to be the identity function.
//
// For regular neurons,
// the location of the destination neuron is projected onto the neurons of the source
// layer. A shape is defined by the radius parameters, and is considered to be either
// rectangular or elliptical (depending on the value of projectRectangular below).
// A radius of 0,0 connects a single source neuron in the source layer to this
// destination neuron. E.g., a radius of 1,1, if projectRectangular is true, connects
// nine neurons from the source layer in a 3x3 block to this destination neuron.
//
// Each Neuron object holds a container of Connection objects for all the source
// inputs to the neuron. Each neuron also holds a container of pointers to Connection
// objects in the forward direction. Those points point to Container objects in
// and owned by neurons in other layers. I.e., the master copies of all connection are
// in the containers of back connections; forward connection pointers refer to
// back connection records in other neurons.
//
void Net::connectNeuron(Layer &layerTo, Layer &fromLayer, Neuron &neuron,
        uint32_t nx, uint32_t ny)
{
    auto &params = layerTo.params;
    uint32_t sizeX = params.size.x;
    uint32_t sizeY = params.size.y;
    assert(sizeX > 0 && sizeY > 0);

    // Calculate the normalized [0..1] coordinates of our neuron:
    float normalizedX = ((float)nx / sizeX) + (1.0 / (2 * sizeX));
    float normalizedY = ((float)ny / sizeY) + (1.0 / (2 * sizeY));

    // Calculate the coords of the nearest neuron in the "from" layer.
    // The calculated coords are relative to the "from" layer:
    uint32_t lfromX = uint32_t(normalizedX * fromLayer.params.size.x); // should we round off instead of round down?
    uint32_t lfromY = uint32_t(normalizedY * fromLayer.params.size.y);

//    info << "our neuron at " << nx << "," << ny << " covers neuron at "
//         << lfromX << "," << lfromY << endl;

    // Calculate the rectangular window into the "from" layer:

    int32_t xmin = 0;
    int32_t xmax = 0;
    int32_t ymin = 0;
    int32_t ymax = 0;

    if (params.isConvolutionFilterLayer) {
        assert(params.convolveMatrix[0].size() > 0);
        assert(params.convolveMatrix[0].size() > 0);

        ymin = lfromY - params.convolveMatrix[0][0].size() / 2;
        ymax = ymin + params.convolveMatrix[0][0].size() - 1;
        xmin = lfromX - params.convolveMatrix[0].size() / 2;
        xmax = xmin + params.convolveMatrix[0].size() - 1;
    } else {
        xmin = lfromX - params.radius.x;
        xmax = lfromX + params.radius.x;
        ymin = lfromY - params.radius.y;
        ymax = lfromY + params.radius.y;
    }

    // Clip to the layer boundaries:

    if (xmin < 0) xmin = 0;
    if (xmin >= (int32_t)fromLayer.params.size.x) xmin = fromLayer.params.size.x - 1;
    if (ymin < 0) ymin = 0;
    if (ymin >= (int32_t)fromLayer.params.size.y) ymin = fromLayer.params.size.y - 1;
    if (xmax < 0) xmax = 0;
    if (xmax >= (int32_t)fromLayer.params.size.x) xmax = fromLayer.params.size.x - 1;
    if (ymax < 0) ymax = 0;
    if (ymax >= (int32_t)fromLayer.params.size.y) ymax = fromLayer.params.size.y - 1;

    // Now (xmin,xmax,ymin,ymax) defines a rectangular subset of neurons in a previous layer.
    // We'll make a connection from each of those neurons in the previous layer to our
    // neuron in the current layer.

    // We will also check for and avoid duplicate connections. Duplicates are mostly harmless,
    // but unnecessary. Duplicate connections can be formed when the same layer name appears
    // more than once in the topology config file with the same "from" layer if the projected
    // rectangular or elliptical areas on the source layer overlap.

    float xcenter = ((float)xmin + (float)xmax) / 2.0;
    float ycenter = ((float)ymin + (float)ymax) / 2.0;
    uint32_t maxNumSourceNeurons = ((xmax - xmin) + 1) * ((ymax - ymin) + 1);

    for (int32_t y = ymin; y <= ymax; ++y) {
        for (int32_t x = xmin; x <= xmax; ++x) {
            if (!params.isConvolutionFilterLayer && !projectRectangular && elliptDist(xcenter - x, ycenter - y,
                                                  params.radius.x, params.radius.y) >= 1.0) {
                continue; // Skip this location, it's outside the ellipse
            }

            if (params.isConvolutionFilterLayer && params.convolveMatrix[0][x - xmin][y - ymin] == 0.0) {
                // Skip this connection because the convolve matrix weight is zero:
                continue;
            }

            Neuron &fromNeuron = fromLayer.neurons[flattenDXY(0, x, y, fromLayer.params.size)];

            bool duplicate = false;
            if (neuron.sourceNeurons.find(&fromNeuron) != neuron.sourceNeurons.end()) {
                duplicate = true;
                break; // Skip this connection, proceed to the next
            }

            if (!duplicate) {
                // Add a new Connection record to the main container of connections:
                connections.push_back(Connection(fromNeuron, neuron));
                int connectionIdx = connections.size() - 1;  //    and get its index,
                ++totalNumberConnections;

                // Initialize the weight of the connection:
                if (params.isConvolutionFilterLayer) {
                    connections.back().weight = params.convolveMatrix[0][x - xmin][y - ymin];
                } else {
                    //connections.back().weight = (randomFloat() - 0.5) / maxNumSourceNeurons;
                    connections.back().weight = ((randomFloat() * 2) - 1.0) / sqrt(maxNumSourceNeurons);
                }

                // Record the back connection index at the destination neuron:
                neuron.backConnectionsIndices.push_back(connectionIdx);

                // Remember the source neuron for detecting duplicate connections:
                neuron.sourceNeurons.insert(&fromNeuron);

                // Record the Connection index at the source neuron:
                uint32_t flatIdxFrom = flattenDXY(0, x, y, fromLayer.params.size);
                fromLayer.neurons[flatIdxFrom].forwardConnectionsIndices.push_back(
                           connectionIdx);

//                info << "    connect from layer " << fromLayer.params.layerName
//                     << " at " << x << "," << y
//                     << " to " << nx << "," << ny << endl;
            }
        }
    }
}


// Add a weighted bias input, modeled as a back-connection to a fake neuron:
//
void Net::connectBias(Neuron &neuron)
{
    // Create a new Connection record and get its index:
    connections.push_back(Connection(bias, neuron));
    uint32_t connectionIdx = connections.size() - 1;

    Connection &c = connections.back();
    c.weight = randomFloat() - 0.5;
    c.deltaWeight = 0.0;

    // Record the back connection with the destination neuron:
    neuron.backConnectionsIndices.push_back(connectionIdx);

    ++totalNumberConnections;

    // Record the forward connection with the fake bias neuron:
    bias.forwardConnectionsIndices.push_back(connectionIdx);
}


// Returns layer index if found, else returns -1
//
int32_t Net::getLayerNumberFromName(const string &name) const
{
    auto it = std::find_if(layers.begin(), layers.end(), [&name](Layer layer) {
                      return layer.params.layerName == name; });

    return it == layers.end() ? -1 : it - layers.begin();
}


// Create neurons and connect them. For the input layer, there are no incoming
// connections and radius doesn't apply. Calling this function with layerFrom == layerTo
// indicates an input layer.
//
void Net::createNeurons(Layer &layerTo, Layer &layerFrom)
{
    // Reserve enough space in layer.neurons to prevent reallocation (so that
    // we can form stable references to neurons):

    layerTo.neurons.reserve(layerTo.params.size.x * layerTo.params.size.y);

    for (uint32_t ny = 0; ny < layerTo.params.size.y; ++ny) {

        info << "\r" << ny << "/" << layerTo.params.size.y << std::flush; // Progress indicator

        for (uint32_t nx = 0; nx < layerTo.params.size.x; ++nx) {
            // When we create a neuron, we have to give it a pointer to the
            // start of the array of Connection objects:
            layerTo.neurons.push_back(&layerTo);
            Neuron &neuron = layerTo.neurons.back(); // Make a more convenient name
            ++totalNumberNeurons;

            // If layerFrom is layerTo, it means we're making input neurons
            // that have no input connections to the neurons. Else, we must make connections
            // to the source neurons and, for classic neurons, to a bias input:

            if (&layerFrom != &layerTo) {
                connectNeuron(layerTo, layerFrom, neuron, nx, ny);
                if (!layerTo.params.isConvolutionFilterLayer) {
                    connectBias(neuron);
                }
            }
        }
    }

    info << endl; // End the progress indicator
}


/*
Convolution filter matrix example formats:
{0, 1,2}
{ {0,1,2}, {1,2,1}, {0, 1, 0}}
*/
convolveMatrix_t Net::parseMatrixSpec(std::istringstream &ss)
{
    char c;
    enum state_t { INIT, LEFTBRACE, RIGHTBRACE, COMMA, NUM };
    enum action_t { SKIP, ILL, PLINC, PLDECX, STONYINC, STONXINC, ACCUM };
    state_t lastState = INIT;
    state_t newState = INIT;
    int braceLevel = 0;
    vector<float> row;
    vector<vector<float>> mat;
    float num = 0.0;

    action_t table[5][5] = {
      /*                 INIT LEFTBRACE RIGHTBRACE COMMA     NUM  */
      /* INIT */       { ILL, PLINC,    ILL,       ILL,      ILL   },
      /* LEFTBRACE */  { ILL, PLINC,    ILL,       ILL,      ACCUM },
      /* RIGHTBRACE */ { ILL, ILL,      PLDECX,    SKIP,     ILL   },
      /* COMMA */      { ILL, PLINC,    ILL,       ILL,      ACCUM },
      /* DIGIT */      { ILL, ILL,      STONYINC,  STONXINC, ACCUM },
    };

    bool done = false;
    while (!done && ss) {
        ss >> c;
        if (isspace(c)) {
            continue;
        } else if (c == '{') {
            newState = LEFTBRACE;
        } else if (c == '}') {
            newState = RIGHTBRACE;
        } else if (c == ',') {
            newState = COMMA;
        } else if (c == '-' || c == '+' || c == '.' || isdigit(c)) {
            newState = NUM;
        } else {
            err << "Internal error in parsing convolve matrix spec" << endl;
            throw exceptionRuntime();
        }

        action_t action = table[lastState][newState];

        switch(action) {
        case SKIP:
            break;
        case ILL:
            err << "Error in convolve matrix spec" << endl;
            throw exceptionConfigFile();
            break;
        case PLINC:
            ++braceLevel;
            break;
        case PLDECX:
            --braceLevel;
            if (braceLevel != 0) {
                err << "Error in convolve matrix spec" << endl;
                throw exceptionConfigFile();
            }
            done = true;
            break;
        case STONYINC:
            row.push_back(num); // Add the element to the row
            mat.push_back(row); // Add the row to the matrix
            row.clear();
            num = 0.0; // Start a new number after this
            if (--braceLevel == 0) {
                done = true;
            }
            break;
        case STONXINC:
            row.push_back(num); // Add the element to the row
            num = 0.0; // Start a new number after this
            break;
        case ACCUM:
            // We've got the first char of the number in c, which can be -, +, ., or a digit.
            // Now gather the rest of the numeric string:
            string numstr;
            numstr.clear();
            numstr.push_back(c);
            while (ss.peek() == '.' || isdigit(ss.peek())) {
                char cc;
                ss >> cc;
                numstr.push_back(cc);
            }
            num = strtod(numstr.c_str(), NULL);
            break;
        }

        lastState = newState;
    }

    // Transpose the matrix so that we can access elements as [x][y]
    // This matters only if the matrix is asymmetric. While we're doing
    // this, we'll check that all rows have the same size, and we'll
    // record the sum of the weights.

    convolveMatrix_t convMat;
    unsigned firstRowSize = 0;

    for (unsigned x = 0; x < mat.size(); ++x) {
        if (x == 0) {
            firstRowSize = mat[x].size(); // Remember the first row size
        } else if (mat[x].size() != firstRowSize) {
            err << "Error: in convolution filter matrix in topology config file, inconsistent matrix row size" << endl;
            throw exceptionConfigFile();
        }
    }
    for (unsigned y = 0; y < firstRowSize; ++y) {
        convMat.push_back(vector<float>());
        for (unsigned x = 0; x < mat.size(); ++x) {
            convMat.back().push_back(mat[x][y]);
        }
    }

    return convMat;
}


// It's possible that some internal neurons don't feed any other neurons.
// That's not a fatal error, but it's probably due to an unintentional mistake
// in defining the net topology. Here we will find and report all neurons with
// no forward connections so that the human can fix the topology configuration
// if needed:
void Net::reportUnconnectedNeurons(void)
{
    warn << "\nChecking for neurons with no sinks:" << endl;

    // Loop through all layers except the output layer, looking for unconnected neurons:
    uint32_t neuronsWithNoSink = 0;
    for (uint32_t lidx = 0; lidx < layers.size() - 1; ++lidx) {
        Layer const &layer = layers[lidx];
        for (auto const &neuron : layer.neurons) {
            if (neuron.forwardConnectionsIndices.size() == 0) {
                ++neuronsWithNoSink;
                warn << "  neuron(" << &neuron << ") on " << layer.params.layerName
                     << endl;
            }
        }
    }
}


// Returns true if the neural net was successfully created and connected. Returns
// false for any error. See the GitHub wiki (https://github.com/davidrmiller/neural2d)
// for more information about the format of the topology config file.
// Throws an exception for any error.
//
void Net::configureNetwork(vector<topologyConfigSpec_t> allLayerSpecs, const string configFilename)
{
    uint32_t numNeurons = 0;

    // We want to pre-allocate the .layers member so that we can form persistent
    // references to individual layers. We could do this more exactly, but a safe
    // heuristic is to allocate as many layers as elements in the config spec array:
    layers.reserve(allLayerSpecs.size());

    for (topologyConfigSpec_t &spec : allLayerSpecs) {
        // Find indices of existing source and dest layers, or -1 if not found:
        int32_t previouslyDefinedLayerNumSameName = getLayerNumberFromName(spec.layerParams.layerName);
        int32_t layerNumFrom = getLayerNumberFromName(spec.fromLayerName); // input layer will return -1

        // If the layer of this name does not already exist, create it:
        if (previouslyDefinedLayerNumSameName == -1) {
            // Create a new layer
            Layer &newLayer = createLayer(spec.layerParams);

            // Create neurons and connect them:

            info << "Creating layer " << spec.layerParams.layerName << ", one moment..." << endl;
            if (newLayer.params.layerName == "input") {
                createNeurons(newLayer, newLayer); // Input layer has no back connections
            } else {
                createNeurons(newLayer, layers[layerNumFrom]); // Also connects them
            }

            numNeurons += newLayer.params.size.x * newLayer.params.size.y;
        } else {
            // Layer already exists, add connections to it.
            // "input" layer will never take this path.
            previouslyDefinedLayerNumSameName = getLayerNumberFromName(spec.layerParams.layerName);
            Layer &layerTo = layers[previouslyDefinedLayerNumSameName]; // A more convenient name

            // Add more connections to the existing neurons in this layer:

            bool ok = addToLayer(layerTo, layers[layerNumFrom]);
            if (!ok) {
                err << "Error in " << configFilename << ", layer \'" << layerTo.params.layerName << "\'" << endl;
                throw exceptionConfigFile();
            }
        }
    }
}

void Net::parseConfigFile(const string &configFilename)
{
    if (!isFileExists(configFilename)) {
        err << "Error reading topology file \'" << configFilename << "\'" << endl;
        throw exceptionConfigFile();
    }

    std::ifstream cfg(configFilename);
    if (!cfg) {
        err << "Error reading topology file \'" << configFilename << "\'" << endl;
        throw exceptionConfigFile();
    }

    configureNetwork(parseTopologyConfig(cfg), configFilename);

    // Record the location of the connections container in the Layers:
    for_each(layers.begin(), layers.end(), [this](Layer &layer) {
             layer.pConnections = &connections; });

    reportUnconnectedNeurons();
}


#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)

// This is the interface between the Net class and the WebServer class. Here
// we construct a string containing lines of JavaScript variable assignments that
// the web server can insert into the HTTP HTML response page.
//
void Net::makeParameterBlock(string &s)
{
    s = "";

    // isRunning=0|1

    s.append("isRunning=");
    s.append(isRunning ? "1" : "0");
    s.append(";\r\n");

    // targetOutputsDefined=0|1

    s.append("targetOutputsDefined=");
    if (sampleSet.samples[0].targetVals.size() > 0) {
        s.append("1;\r\n");
    } else {
        s.append("0;\r\n");
    }

    // runMode="runOnce|runRepeat|runRepeatShuffle"

    if (repeatInputSamples && shuffleInputSamples) {
        s.append("runMode=\"runRepeatShuffle\";\r\n");
    } else if (repeatInputSamples && !shuffleInputSamples) {
        s.append("runMode=\"runRepeat\";\r\n");
    } else {
        s.append("runMode=\"runOnce\";\r\n");
    }

    // train=0|1

    if (enableBackPropTraining) {
        s.append("train=1;\r\n");
    } else {
        s.append("train=0;\r\n");
    }

    // stopError=float

    s.append("stopError=" + to_string(doneErrorThreshold) + ";\r\n");

    // channel=R|G|B|BW

    string channel;
    switch(layers[0].params.channel) {
        case NNet::R: channel = "R"; break;
        case NNet::G: channel = "G"; break;
        case NNet::B: channel = "B"; break;
        case NNet::BW: channel = "BW"; break;
    }

    s.append("channel=\"" + channel + "\";\r\n");

    // eta=float

    s.append("eta=" + to_string(eta) + ";\r\n");

    // dynamicEta=0|1

    s.append("dynamicEta=");
    s.append(dynamicEtaAdjust ? "1" : "0");
    s.append(";\r\n");

    // alpha=float

    s.append("alpha=" + to_string(alpha) + ";\r\n");

    // lambda=float

    s.append("lambda=" + to_string(lambda) + ";\r\n");

    // reportEveryNth=int

    s.append("reportEveryNth=" + to_string(reportEveryNth) + ";\r\n");

    // smoothingFactor=float

    s.append("smoothingFactor=" + to_string(recentAverageSmoothingFactor) + ";\r\n");

    // weightsFile="text"
    s.append("weightsFile=\"" + weightsFilename + "\";\r\n");

    // portNumber=int
    s.append("portNumber=" + to_string(webServer.portNumber) + ";\r\n");

    return;
}
#endif


#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
// Handler for the optional external controller for the neural2d program.
// This function reads the command file and acts on any commands received.
//
void Net::actOnMessageReceived(Message_t &msg)
{
    string parameterBlock;
    ColorChannel_t newColorChannel = layers[0].params.channel;

    string &line = msg.text;

    //info << "Acting on message: \"" << line << "\"" << endl;

    if (line == "" &&  msg.httpResponseFileDes != -1) {
        makeParameterBlock(parameterBlock);
        webServer.sendHttpResponse(parameterBlock, msg.httpResponseFileDes);
        return;
    }

    std::istringstream ss(line);
    string token;

    ss >> token;

    // trainShadow

    if (token.find("trainShadow=&train=on") == 0) {
        enableBackPropTraining = true;
        info << "Enable backprop training" << endl;
    } else if (token.find("trainShadow=") == 0) {
        enableBackPropTraining = false;
        info << "Disable backprop training" << endl;
    }

    else if (token.find("training=") == 0) {
        enableBackPropTraining = true;
        doneErrorThreshold = 0.01;
        reportEveryNth = 125;
        recentAverageSmoothingFactor = 100;
    } else if (token.find("validate=") == 0) {
        enableBackPropTraining = false;
        doneErrorThreshold = 0.0;
        reportEveryNth = 1;
        recentAverageSmoothingFactor = 1;
    } else if (token.find("trained=") == 0) {
        enableBackPropTraining = false;
        reportEveryNth = 1;
    }

    // stopError

    else if (token.find("stopError=") == 0) {
        doneErrorThreshold = strtod(token.substr(10).c_str(), NULL);
        info << "Pause when error < " << doneErrorThreshold << endl;
    }

    else if (token.find("runOnceShadow=") == 0) {
        repeatInputSamples = false;
        shuffleInputSamples = false;
    } else if (token.find("runRepeatShadow=") == 0) {
        repeatInputSamples = true;
        shuffleInputSamples = false;
    } else if (token.find("runRepeatShuffleShadow=") == 0) {
        repeatInputSamples = true;
        shuffleInputSamples = true;
    }

    // colorchannel
    // If the color channel changes after we have already started caching pixel data,
    // then we need to dump the cached data so that the images will be re-read and
    // converted using the new color channel.

    else if (token.find("channelRShadow=") == 0) {
        newColorChannel = NNet::R;
        info << "Color channel = R" << endl;
    } else if (token.find("channelGShadow=") == 0) {
        newColorChannel = NNet::G;
        info << "Color channel = G" << endl;
    } else if (token.find("channelBShadow=") == 0) {
        newColorChannel = NNet::B;
        info << "Color channel = B" << endl;
    } else if (token.find("channelBWShadow=") == 0) {
        newColorChannel = NNet::BW;
        info << "Color channel = BW" << endl;
    }

    // alpha

    else if (token.find("alpha=") == 0) {
        alpha = strtod(token.substr(6).c_str(), NULL);
        info << "Set alpha=" << alpha << endl;
    }

    else if (token.find("eta=") == 0) {
        eta = strtod(token.substr(4).c_str(), NULL);
        info << "Set eta=" << eta << endl;
    }

    else if (token.find("etaShadow=&dynamicEta=1") == 0) {
        dynamicEtaAdjust = true;
        info << "dynamicEtaAdjust=" << dynamicEtaAdjust << endl;
    }

    else if (token.find("etaShadow=") == 0) {
        dynamicEtaAdjust = false;
        info << "dynamicEtaAdjust=" << dynamicEtaAdjust << endl;
    }

    else if (token.find("lambda=") == 0) {
        lambda = strtod(token.substr(7).c_str(), NULL);
        info << "Set lambda=" << lambda << endl;
    }

    else if (token == "load") {
        ss >> token;
        info << "Load weights from " << token << endl;
        loadWeights(token);
    }

    else if (token.find("pause") == 0) {
        isRunning = false;
        info << "Pause" << endl;
    }

    else if (token.find("reportEveryNth=") == 0) {
        reportEveryNth = strtod(token.substr(15).c_str(), NULL);
        info << "Report everyNth=" << reportEveryNth << endl;
    }

    else if (token.find("smoothingFactor=") == 0) {
        recentAverageSmoothingFactor = strtod(token.substr(16).c_str(), NULL);
        info << "Average window over " << recentAverageSmoothingFactor << endl;
    }

    else if (token.find("weightsFile=") == 0) {
        weightsFilename = token.substr(12);
        sanitizeFilename(weightsFilename);
        info << "weightsFilename = " << weightsFilename << endl;
    }

    else if (token == "run" || token.find("resume") == 0) {
        isRunning = true;
        info << "Resume run" << endl;
    }

    else if (token.find("savew") == 0) {
        info << "Save weights to " << weightsFilename << endl;
        saveWeights(weightsFilename);
    }

    else if (token.find("loadw") == 0) {
        info << "Load weights from " << weightsFilename << endl;
        loadWeights(weightsFilename);
    }

    else if (token == "repeat") {
        ss >> token;
        repeatInputSamples = (token == "True");
        info << "repeatInputSamples=" << repeatInputSamples << endl;
    }

    else if (token == "shuffle") {
        ss >> token;
        shuffleInputSamples = (token == "True");
        info << "shuffleInputSamples=" << shuffleInputSamples << endl;
    }

    // Post processing

    if (newColorChannel != layers[0].params.channel) {
        sampleSet.clearImageCache();
        layers[0].params.channel = newColorChannel;
    }

    // Send the HTTP response:
    // To do: use async() !!!

    makeParameterBlock(parameterBlock);
    webServer.sendHttpResponse(parameterBlock, msg.httpResponseFileDes);
}
#endif


#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
void Net::doCommand()
{
    // Check the web interface:
    do {
        Message_t msg;
        messages.pop(msg);
        if (msg.httpResponseFileDes != -1) {
            actOnMessageReceived(msg);
        }
        if (!isRunning) {
            usleep(100000); // Slow the polling
        }
    } while (!isRunning);
}
#endif


// Calculate a new eta parameter based on the current and last average net error.
//
float Net::adjustedEta(void)
{
    const float thresholdUp = 0.001;       // Ignore error increases less than this magnitude
    const float thresholdDown = 0.01;      // Ignore error decreases less than this magnitude
    const float factorUp = 1.005;          // Factor to incrementally increase eta
    const float factorDown = 0.999;        // Factor to incrementally decrease eta

    if (!dynamicEtaAdjust) {
        return eta;
    }

    assert(thresholdUp > 0.0 && thresholdDown > 0.0 && factorUp >= 1.0 && factorDown >= 0.0 && factorDown <= 1.0);

    float errorGradient = (recentAverageError - lastRecentAverageError) / recentAverageError;
    if (errorGradient > thresholdUp) {
        eta = factorDown * eta;
    } else if (errorGradient < -thresholdDown) {
        eta = factorUp * eta;
    }

    return eta;
}


} // end namespace NNet
