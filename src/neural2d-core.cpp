/*
neural2d-core.cpp
David R. Miller, 2014, 2015
https://github.com/davidrmiller/neural2d

See neural2d.h for more information.
*/

#include <cctype>
#include <cmath>
#include <memory>

// On *nix, we have usleep(), but in Windows, we have to simulate it:
#if defined(WIN32)
    #include <chrono>
    #include <thread>
    #define usleep(usec) std::this_thread::sleep_for(std::chrono::microseconds(usec));
#else
    #include <unistd.h> // For sleep() or usleep()
#endif

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
uint32_t flattenXY(uint32_t x, uint32_t y, uint32_t ySize)
{
    return x * ySize + y;
}

uint32_t flattenXY(uint32_t x, uint32_t y, dxySize size)
{
    return flattenXY(x, y, size.y);
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
float transferFunctionDerivativeTanh(float x) { return 1.0f - tanh(x) * tanh(x); }

// logistic is a sigmoid curve that ranges 0.0 to 1.0:
float transferFunctionLogistic(float x) { return 1.0f / (1.0f + exp(-x)); }
float transferFunctionDerivativeLogistic(float x) { return exp(-x) / pow((exp(-x) + 1.0f), 2.0f); }

// linear is a constant slope; ranges from -inf to +inf:
float transferFunctionLinear(float x) { return x; }
float transferFunctionDerivativeLinear(float x) { return (void)x, 1.0f; }

// ramp is a constant slope between -1 <= x <= 1, zero slope elsewhere; output ranges from -1 to +1:
float transferFunctionRamp(float x)
{
    if (x < -1.0) return -1.0;
    else if (x > 1.0) return 1.0;
    else return x;
}
float transferFunctionDerivativeRamp(float x) { return (x < -1.0f || x > 1.0f) ? 0.0f : 1.0f; }

// gaussian:
float transferFunctionGaussian(float x) { return exp(-((x * x) / 2.0f)); }
float transferFunctionDerivativeGaussian(float x) { return -x * exp(-(x * x) / 2.0f); }

// ReLU:
// This is a softplus approximation of a ReLU:
float transferFunctionReLU(float x) { return log(1.0f + exp(x)); }
float transferFunctionDerivativeReLU(float x) { return 1.0f / (1.0f + exp(-x)); }

// Identity function:
float transferFunctionIdentity(float x) { return x; }
float transferFunctionIdentityDerivative(float x) { return (void)x, 1.0f; }


// ***********************************  Input samples  ***********************************


// Given an integer 8-bit pixel value in the range 0..255, convert that into
// a floating point value that can be input directly into the input layer of
// the neural net. If you change this, also change the inverse function
// networkInputValToPixelRange()
//
float pixelToNetworkInputRange(unsigned val)
{
    // Return a value in the range -1..1:
    return val / 128.0f - 1.0f;

    // Alternatively, return a value in the range 0..1:
    //return val / 256.0f;

    // Alternatively, return a value in the range -0.5..0.5:
    //return val / 256.0f - 0.5f;
}

// If you change this, also change the inverse function
// pixelToNetworkInputRange().
//
unsigned networkInputValToPixelRange(float val)
{
    return min((unsigned)((val + 1.0f) * 128 + 0.5f), 255);

    // Alternatively for a value in the range 0..1:
    //return min((unsigned)(val * 256 + 0.5f), 255);

    // Alternatively for a value in the range -0.5..0.5:
    //return min((unsigned)((val + 0.5f) * 256 + 0.5f), 255);
}


// ***********************************  class Sample  ***********************************


// If the data is available, we'll return it. If this is the first time getData()
// is called for inputs that come from an image, we'll open the image file and
// cache the pixel data in memory. Returns a reference to the container of input data.
//
vector<float> const &Sample::getData(ColorChannel_t channel)
{
   if (data.size() == 0 && imageFilename != "") {
       // Try all the image readers until we find one that succeeds:
       for (auto imageReader : SampleSet::imageReaders) {
           size = imageReader->getData(imageFilename, data, channel);
           if (size.x != 0) {
               break;
           }
       }

       if (size.x == 0) {
           err << "Unsupported image file format in " << imageFilename << std::endl;
           throw exceptionInputSamplesFile();
       }
   }

   // If we get here, we can assume there is something in the .data member

   return data;
}


void Sample::clearImageCache(void)
{
    data.clear();
}


vector<ImageReader *> SampleSet::imageReaders = {
        new ImageReaderDat(),
        new ImageReaderBMP()
};


// ***********************************  class SampleSet  ***********************************


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
// We now honor the directive "path_prefix=", which is a string that gets prepended
// to the front of every filename.
//
void SampleSet::loadSamples(const string &inputFilename)
{
    string line;
    uint32_t lineNum = 0;
    string pathPrefix = "";

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

        if (token.find("path_prefix") == 0) {
            if (token.size() == 11) {
                ss >> delim;
                token += delim; // Attach the equals sign, no spaces
            }
            if (token.size() == 12) {
                string s;
                ss >> s;
                token += s; // Attach the path prefix string, no spaces
            }
            pathPrefix = token.substr(12); // "path_prefix=" is 12 chars
            continue;
        } else if (token == "{") {
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
            sample.imageFilename = pathPrefix + token;
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
void SampleSet::clearImageCache(void)
{
    for (auto &samp : samples) {
        if (samp.imageFilename != "")
            samp.clearImageCache();
    }
}


// ***********************************  struct Connection  ***********************************


Connection::Connection(Neuron &from, Neuron &to)
     : fromNeuron(from), toNeuron(to)
{
    weight = randomFloat() / 2.0f - 1.0f;  // Range -0.25..0.25
    deltaWeight = 0.0f;
    convolveMatrixIndex = 0;
}


// *****************************  class Layer  *****************************


Layer::Layer(const topologyConfigSpec_t &params)
{
    layerName = params.layerName;
    size = params.size;
    isRegularLayer = params.isRegularLayer;
    isConvolutionFilterLayer = params.isConvolutionFilterLayer;
    isConvolutionNetworkLayer = params.isConvolutionNetworkLayer;
    isPoolingLayer = params.isPoolingLayer;
    resolveTransferFunctionName(params.transferFunctionName);
    totalNumberBackConnections = 0;
    projectRectangular = false;
}

void Layer::resolveTransferFunctionName(string const &transferFunctionName)
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
    } else if (transferFunctionName == "relu" || transferFunctionName == "ReLU") {
        tf = transferFunctionReLU;
        tfDerivative = transferFunctionDerivativeReLU;
    } else if (transferFunctionName == "identity") {
        tf = transferFunctionIdentity;
        tfDerivative = transferFunctionIdentityDerivative;
    } else {
        err << "Undefined transfer function: \'" << transferFunctionName << "\'" << endl;
        throw exceptionConfigFile();
    }
}

void Layer::clipToBounds(int32_t &xmin, int32_t &xmax, int32_t &ymin, int32_t &ymax, dxySize &size)
{
    if (xmin < 0) xmin = 0;
    if (xmin >= (int32_t)size.x) xmin = size.x - 1;
    if (ymin < 0) ymin = 0;
    if (ymin >= (int32_t)size.y) ymin = size.y - 1;
    if (xmax < 0) xmax = 0;
    if (xmax >= (int32_t)size.x) xmax = size.x - 1;
    if (ymax < 0) ymax = 0;
    if (ymax >= (int32_t)size.y) ymax = size.y - 1;
}

void Layer::saveWeights(std::ofstream &) { }

void Layer::loadWeights(std::ifstream &) { }

void Layer::calcGradients(const vector<float> &targetVals)
{
    if (layerName == "output") {
        for (uint32_t n = 0; n < neurons[0].size(); ++n) {
            neurons[0][n].calcOutputGradients(targetVals[n], tfDerivative);
        }
    } else {
        assert(layerName != "input");
        for (auto &plane : neurons) {
            for (auto &neuron : plane) {
                neuron.calcHiddenGradients(*this);
            }
        }
    }
}

void Layer::updateWeights(float eta, float alpha)
{
    for (auto &plane : neurons) {
        for (auto &neuron : plane) {
            neuron.updateInputWeights(eta, alpha, pConnections);
        }
    }
}

void Layer::connectLayers(Layer &layerFrom)
{
    for (uint32_t destDepth = 0; destDepth < size.depth; ++destDepth) {
        for (uint32_t destX = 0; destX < size.x; ++destX) {
            for (uint32_t destY = 0; destY < size.y; ++destY) {
                auto &toNeuron = neurons[destDepth][flattenXY(destX, destY, size)];
                connectOneNeuronAllDepths(layerFrom, toNeuron, destDepth, destX, destY);
            }
        }
    }
}


// Add a weighted bias input, modeled as a back-connection to a fake neuron:
//
void Layer::connectBiasToAllNeuronsAllDepths(Neuron &bias)
{
    for (auto &plane : neurons) {
        for (auto &neuron : plane) {
            // Create a new Connection record and get its index:
            (*pConnections).push_back(Connection(bias, neuron));
            uint32_t connectionIdx = (*pConnections).size() - 1;

            Connection &c = (*pConnections).back();
            c.weight = randomFloat() / (size.x * size.y);
            c.deltaWeight = 0.0;

            // Record the back connection with the destination neuron:
            neuron.backConnectionsIndices.push_back(connectionIdx);

            ++totalNumberBackConnections;

            // Record the forward connection with the fake bias neuron:
            bias.forwardConnectionsIndices.push_back(connectionIdx);
        }
    }
}

void Layer::connectOneNeuronAllDepths(Layer &fromLayer, Neuron &toNeuron,
        uint32_t destDepth, uint32_t destX, uint32_t destY)
{
    auto &layerTo = *this;
    assert(size.x > 0 && size.y > 0);

    // Calculate the normalized [0..1] coordinates of our neuron:
    float normalizedX = ((float)destX / size.x) + (1.0f / (2 * size.x));
    float normalizedY = ((float)destY / size.y) + (1.0f / (2 * size.y));

    // Calculate the coords of the nearest neuron in the "from" layer.
    // The calculated coords are relative to the "from" layer:
    uint32_t lfromX = uint32_t(normalizedX * fromLayer.size.x); // should we round off instead of round down?
    uint32_t lfromY = uint32_t(normalizedY * fromLayer.size.y);

//    info << "our neuron at " << destX << "," << ny << " covers neuron at "
//         << lfromX << "," << lfromY << endl;

    // Calculate the rectangular window into the "from" layer:

    int32_t xmin, xmax, ymin, ymax;

    if (isRegularLayer) {
        xmin = lfromX - layerTo.radius.x;
        xmax = lfromX + layerTo.radius.x;
        ymin = lfromY - layerTo.radius.y;
        ymax = lfromY + layerTo.radius.y;
        clipToBounds(xmin, xmax, ymin, ymax, fromLayer.size);
    }
    else if (isConvolutionFilterLayer || isConvolutionNetworkLayer) {
        ymin = lfromY - layerTo.kernelSize.y / 2;
        ymax = ymin   + layerTo.kernelSize.y - 1;
        xmin = lfromX - layerTo.kernelSize.x / 2;
        xmax = xmin   + layerTo.kernelSize.x - 1;
    }
    else { // isPoolingLayer
        ymin = lfromY - layerTo.poolSize.y / 2;
        ymax = ymin   + layerTo.poolSize.y - 1;
        xmin = lfromX - layerTo.poolSize.x / 2;
        xmax = xmin   + layerTo.poolSize.x - 1;
    }

    // Now (xmin,xmax,ymin,ymax) defines a rectangular subset of neurons in a previous layer.
    // We'll make a connection from each of those neurons in the previous layer to our
    // neuron in the current layer.

    // We will also check for and avoid duplicate connections. Duplicates are mostly harmless,
    // but unnecessary. Duplicate connections can be formed when the same layer name appears
    // more than once in the topology config file with the same "from" layer if the projected
    // rectangular or elliptical areas on the source layer overlap.

    float srcCenterX = ((float)xmin + (float)xmax) / 2.0f; // for elliptical calculations
    float srcCenterY = ((float)ymin + (float)ymax) / 2.0f; // for elliptical calculations

    uint32_t maxNumSourceNeurons = ((xmax - xmin) + 1) * ((ymax - ymin) + 1); // for heuristic weight initializations

    // The way we connect to the source layer depends on the depth of the source layer:
    //
    //   src depth == my depth  -- connect only to same depth in source
    //   src depth != my depth  -- fully connect to all depths

    uint32_t sourceDepthMin = 0;
    uint32_t sourceDepthMax = 0;

    if (fromLayer.size.depth == layerTo.size.depth) {
        sourceDepthMin = sourceDepthMax = destDepth;
    } else {
        sourceDepthMin = 0;
        sourceDepthMax = fromLayer.size.depth - 1;
    }

    for (int32_t srcX = xmin; srcX <= xmax; ++srcX) {
        for (int32_t srcY = ymin; srcY <= ymax; ++srcY) {
            // Some layer types may allow either a rectangular or elliptical project pattern:
            if (isRegularLayer) {
                if (!projectRectangular
                            && elliptDist(srcCenterX - (float)srcX, 
                                          srcCenterY - (float)srcY, 
                                          (float)layerTo.radius.x, (float)layerTo.radius.y) >= 1.0f) {
                    continue; // Skip this location, it's outside the ellipse
                }
            } else if (!isRegularLayer) {
                if (srcX < 0 || srcY < 0 || srcX >= (int32_t)fromLayer.size.x || srcY >= (int32_t)fromLayer.size.y) {
                    continue; // Skip, out of bounds
                }
            }

            for (uint32_t sourceDepth = sourceDepthMin; sourceDepth <= sourceDepthMax; ++sourceDepth) {
                Neuron &fromNeuron = fromLayer.neurons[sourceDepth][flattenXY(srcX, srcY, fromLayer.size)];

                // Test for a duplicate connection and connect if needed:
                if (toNeuron.sourceNeurons.find(&fromNeuron) == toNeuron.sourceNeurons.end()) {
                    // Add a new Connection record to the main container of connections:
                    (*pConnections).push_back(Connection(fromNeuron, toNeuron));
                    int connectionIdx = (*pConnections).size() - 1;  //  and get its index
                    ++totalNumberBackConnections;

                    if (isRegularLayer) {
                        // Initialize the weight of the connection: it was already given a random value
                        // when constructed, but we can adjust or replace that here given our more
                        // complete knowledge of the layer topology:
                        (*pConnections).back().weight = (float)(((randomFloat() * 2) - 1.0f) / sqrt(maxNumSourceNeurons));
                    } else if (isConvolutionFilterLayer || isConvolutionNetworkLayer) {
                        // Remember which kernel element goes with this connection:
                        uint32_t matIdx = flattenXY(srcX - xmin, srcY - ymin, kernelSize.y);
                        (*pConnections).back().convolveMatrixIndex = matIdx;
                    } else {
                        // pooling does not use weights
                    }

                    // Record the back connection index at the destination neuron:
                    toNeuron.backConnectionsIndices.push_back(connectionIdx);

                    // Remember the source neuron for detecting duplicate connections:
                    toNeuron.sourceNeurons.insert(&fromNeuron);

                    // Record the Connection index at the source neuron:
                    fromNeuron.forwardConnectionsIndices.push_back(connectionIdx);
                }
            }
        }
    }
}


void Layer::debugShow(bool)
{
}

LayerConvolution::LayerConvolution(const topologyConfigSpec_t &params) : Layer(params)
{
    kernelSize = params.kernelSize;
    flatConvolveMatrix.clear();
    flatConvolveMatrix = params.flatConvolveMatrix;
    flatConvolveGradients.clear();
    flatConvolveGradients.assign(size.depth, vector<float>(kernelSize.x * kernelSize.y));
    flatDeltaWeights.clear();
    flatDeltaWeights.assign(size.depth, vector<float>(kernelSize.x * kernelSize.y));
}

void LayerConvolution::feedForward()
{
    for (uint32_t depthIdx = 0; depthIdx < size.depth; ++depthIdx) {
        for (uint32_t x = 0; x < size.x; ++x) {
            for (uint32_t y = 0; y < size.y; ++y) {
                auto &neuron = neurons[depthIdx][flattenXY(x, y, size)];
                neuron.feedForwardConvolution(depthIdx, this);
            }
        }
    }
}


LayerConvolutionFilter::LayerConvolutionFilter(const topologyConfigSpec_t &params) : LayerConvolution(params)
{
    flatConvolveMatrix = params.flatConvolveMatrix;
}

void LayerConvolutionFilter::debugShow(bool)
{
    info << layerName << ": 1*" << size.x << "x" << size.y
         << " = " << neurons.size() * neurons[0].size() << " neurons"
         << " convolution filter " << kernelSize.x << "x" << kernelSize.y << endl;
}


LayerConvolutionNetwork::LayerConvolutionNetwork(const topologyConfigSpec_t &params) : LayerConvolution(params)
{

}

void LayerConvolutionNetwork::calcGradients(const vector<float> &targetVals)
{
    (void)targetVals;

    assert(flatConvolveGradients[0][0] == 0.0);

    for (uint32_t depth = 0; depth < size.depth; ++depth) {
        for (auto &neuron : neurons[depth]) {
            neuron.calcHiddenGradientsConvolution(depth, *this);
        }
    }
}

void LayerConvolutionNetwork::updateWeights(float eta, float alpha)
{
    for (uint32_t depth = 0; depth < size.depth; ++depth) {
        auto &plane = neurons[depth];
        for (auto &neuron : plane) {
            neuron.updateInputWeightsConvolution(depth, eta, alpha, *this);
        }

        for (size_t wIdx = 0; wIdx < flatConvolveMatrix[depth].size(); ++wIdx) {
            flatConvolveMatrix[depth][wIdx] += flatDeltaWeights[depth][wIdx];
            // We can clear the flatDeltaWeights and flatConvolveGradients items now,
            // we're done with them, then we don't have to make a special loop to clear
            // them at the beginning of backProp:
            flatDeltaWeights[depth][wIdx] = 0;
            flatConvolveGradients[depth][wIdx] = 0;
        }
    }
}

void LayerConvolutionNetwork::saveWeights(std::ofstream &file)
{
    for (auto const &kernelInstance : flatConvolveMatrix) {
        for (auto weight : kernelInstance) {
            file << weight << endl;
        }
    }
}

void LayerConvolutionNetwork::loadWeights(std::ifstream &file)
{
    for (auto &kernelInstance : flatConvolveMatrix) {
        for (auto &weight : kernelInstance) {
            file >> weight;
        }
    }
}

void LayerConvolutionNetwork::debugShow(bool)
{
    info << layerName << ": " << size.depth << "*" << size.x << "x" << size.y
         << " = " << neurons.size() * neurons[0].size() << " neurons, convolution network "
         << kernelSize.x << "x" << kernelSize.y << " kernels" << endl;
}


LayerPooling::LayerPooling(const topologyConfigSpec_t &params) : Layer(params)
{
    poolMethod = params.poolMethod;
    poolSize = params.poolSize;
}

void LayerPooling::feedForward()
{
    for (auto &plane : neurons) {
        for (auto &neuron : plane) {
            neuron.feedForwardPooling(this);
        }
    }
}


void LayerPooling::updateWeights(float, float)
{
}

void LayerPooling::debugShow(bool)
{
    info << layerName << ": " << size.depth << "*" << size.x << "x" << size.y
         << " = " << neurons.size() * neurons[0].size() << " neurons, pool "
         << (poolMethod == POOL_MAX ? "max" : "avg")
         << " " << kernelSize.x << "x" << kernelSize.y << endl;
}


LayerRegular::LayerRegular(const topologyConfigSpec_t &params) : Layer(params)
{
    channel = params.channel;

    if (params.radiusSpecified) {
        radius = params.radius;
    } else {
        radius.x = radius.y = 999999;
    }
}

void LayerRegular::feedForward()
{
    for (auto &plane : neurons) {
        for (auto &neuron : plane) {
            neuron.feedForward(this);
        }
    }
}

void LayerRegular::saveWeights(std::ofstream &file)
{
    for (auto const &plane : neurons) {
        for (auto const &neuron : plane) {
            for (auto idx : neuron.backConnectionsIndices) {
                const Connection &conn = (*pConnections)[idx];
                file << conn.weight << endl;
            }
        }
    }
}

void LayerRegular::loadWeights(std::ifstream &file)
{
    for (auto const &plane : neurons) {
        for (auto const &neuron : plane) {
            for (auto idx : neuron.backConnectionsIndices) {
                Connection &conn = (*pConnections)[idx];
                file >> conn.weight;
            }
        }
    }
}

void LayerRegular::updateWeights(float eta, float alpha)
{
    assert(size.depth == 1);
    for (auto &neuron : neurons[0]) {
        neuron.updateInputWeights(eta, alpha, pConnections);
    }
}

void LayerRegular::debugShow(bool details)
{
    uint32_t numFwdConnections;
    uint32_t numBackConnections;
    auto const &l = *this; // A more convenient name

//    info << "Layer '" << l.layerName << "' has " << l.neurons.size() * l.neurons[0].size()
//         << " neurons arranged in " << l.size.x << "x" << l.size.y
//         << " depth " << l.size.depth << ":" << endl;

    info << l.layerName << ": " << l.size.depth << "*" << l.size.x << "x" << l.size.y
         << " = " << l.neurons.size() * l.neurons[0].size() << " neurons";

    for (size_t depth = 0; depth < l.size.depth; ++depth) {
        numFwdConnections = 0;
        numBackConnections = 0;

        for (auto const &n : l.neurons[depth]) {
            if (details) {
                info << "  neuron(" << &n << ")" << " output: " << n.output << endl;
            }

            numFwdConnections += n.forwardConnectionsIndices.size();
            numBackConnections += n.backConnectionsIndices.size(); // Includes the bias connection

            if (details && n.forwardConnectionsIndices.size() > 0) {
                info << "    Fwd connections:" << endl;
                for (auto idx : n.forwardConnectionsIndices) {
                    Connection const &pc = (*pConnections)[idx];
                    info << "      conn(" << &pc << ") pFrom=" << &pc.fromNeuron
                         << ", pTo=" << &pc.toNeuron
                         << ", w,dw=" << pc.weight << ", " << pc.deltaWeight
                         << endl;
                }
            }

            if (details && n.backConnectionsIndices.size() > 0) {
                info << "    Back connections (incl. bias):" << endl;
                for (auto idx : n.backConnectionsIndices) {
                    Connection const &c = (*pConnections)[idx];
                    info << "      conn(" << &c << ") pFrom=" << &c.fromNeuron
                         << ", pTo=" << &c.toNeuron
                         << ", w=" << c.weight

                         << endl;
                    assert(&c.toNeuron == &n);
                }
            }
        }

        if (!details) {
            info << "\t, " << numBackConnections << " back, "
                 << numFwdConnections << " forward connections";
        }

        info << endl;
    }
}


// ***********************************  class Neuron  ***********************************


Neuron::Neuron()
{
    output = randomFloat() - 0.5f;
    gradient = 0.0f;
    backConnectionsIndices.clear();
    forwardConnectionsIndices.clear();
    sourceNeurons.clear();
}


// The error gradient of an output-layer neuron is equal to the target (desired)
// value minus the computed output value, times the derivative of
// the output-layer activation function evaluated at the computed output value.
//
void Neuron::calcOutputGradients(float targetVal, transferFunction_t tfDerivative)
{
    float delta = targetVal - output;
    gradient = delta * tfDerivative(output);
}


// The error gradient of a hidden-layer neuron is equal to the derivative
// of the activation function of the hidden layer evaluated at the
// local output of the neuron times the sum of the product of
// the primary outputs times their associated hidden-to-output weights.
//
void Neuron::calcHiddenGradients(Layer &myLayer)
{
    float dow = sumDOW_nextLayer(myLayer.pConnections);
    gradient = dow * myLayer.tfDerivative(output);
}


// Special for convolution layers where the weights are stored in the Layer
// object instead of in the Connection records:
//
void Neuron::calcHiddenGradientsConvolution(uint32_t depth, Layer &myLayer)
{
    float sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (auto idx : forwardConnectionsIndices) {
        auto const &conn = (*myLayer.pConnections)[idx];

        // Is the following correct? !!!
        sum += myLayer.flatConvolveMatrix[depth][conn.convolveMatrixIndex]
             * conn.toNeuron.gradient;
    }

    // The individual neuron's gradient:
    gradient = sum * myLayer.tfDerivative(output);
}


// To do: add commentary!!!
//
float Neuron::sumDOW_nextLayer(vector<Connection> *pConnections) const
{
    float sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (auto idx : forwardConnectionsIndices) {
        const Connection &conn = (*pConnections)[idx];

        sum += conn.weight * conn.toNeuron.gradient;
    }

    return sum;
}


void Neuron::updateInputWeights(float eta, float alpha, vector<Connection> *pConnections)
{
    // The weights to be updated are the weights from the neurons in the
    // preceding layer (the source layer) to this neuron:

    for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
        auto idx = backConnectionsIndices[i];
        Connection &conn = (*pConnections)[idx];

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

void Neuron::updateInputWeightsConvolution(uint32_t depth, float eta, float alpha, Layer &myLayer)
{
    // For convolution network layers, the weights to be updated are stored in
    // the Layer object, not the Connection records.

    for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
        auto idx = backConnectionsIndices[i];
        auto const &conn = (*myLayer.pConnections)[idx];
        auto const &fromNeuron = conn.fromNeuron;
        auto &deltaWeight = myLayer.flatDeltaWeights[depth][conn.convolveMatrixIndex];
        auto &gradient    = myLayer.flatConvolveGradients[depth][conn.convolveMatrixIndex];

        float oldDeltaWeight = deltaWeight;

        float newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * fromNeuron.output
                * gradient
                // Add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        deltaWeight += newDeltaWeight; // Just accumulate for now
    }
}


// To feed forward an individual neuron, we'll sum the weighted inputs, then pass that
// sum through the transfer function. This version is for regular layers where the
// weights are stored in the Connection records.
//
void Neuron::feedForward(Layer *pMyLayer)
{
    float sum = 0.0;

    // Sum the neuron's inputs:
    for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
        size_t idx = backConnectionsIndices[i];
        const Connection &conn = (*pMyLayer->pConnections)[idx];

        sum += conn.fromNeuron.output * conn.weight;
    }

    // Shape the output by passing it through the transfer function:
    this->output = pMyLayer->tf(sum);
}

// This version is for convolution network layers where the weights are stored in the
// Layer->flatConvolveMatrix[][] object instead of in the Connection records:
//
void Neuron::feedForwardConvolution(uint32_t depth, Layer *pMyLayer)
{
    output = 0.0;

    // Sum the neuron's inputs:
    for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
        auto idx = backConnectionsIndices[i];
        auto const &conn = (*pMyLayer->pConnections)[idx];
        auto &sourceNeuron = conn.fromNeuron;
        auto kernelElement = pMyLayer->flatConvolveMatrix[depth][conn.convolveMatrixIndex];
        output += sourceNeuron.output * kernelElement;
    }

    // This allows a tf parameter on regular and convolution network layers.
    // Should we also allow tf on convolution filter layers and pooling layers?
    if (!pMyLayer->isConvolutionFilterLayer && !pMyLayer->isPoolingLayer) {
        // Shape the output by passing it through the transfer function:
        output = pMyLayer->tf(output);
    }
}

void Neuron::feedForwardPooling(Layer *pMyLayer)
{
    output = -9999.;

    if (pMyLayer->poolMethod == NNet::POOL_MAX) {
        // Find the max:
        for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
            auto idx = backConnectionsIndices[i];
            auto const &conn = (*pMyLayer->pConnections)[idx];
            auto &sourceNeuron = conn.fromNeuron;
            if (sourceNeuron.output > output) {
                output = sourceNeuron.output;
            }
        }
    } else if (pMyLayer->poolMethod == NNet::POOL_AVG) {
        // Find the average:
        float sum = 0.0;
        for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
            auto idx = backConnectionsIndices[i];
            auto const &conn = (*pMyLayer->pConnections)[idx];
            sum += conn.fromNeuron.output;
        }
        output = sum / backConnectionsIndices.size();
    } else {
        // An unspecified pooling operator is a silent no-op.
    }
}


// ***********************************  class Net  ***********************************


// The second ctor parameter is used by the unit tests to force the webserver off
// even when it was compiled in. Typically you can use preprocessor define -DDISABLE_WEBSERVER
// to prevent the webserver code from being compiled.
//
Net::Net(const string &topologyFilename, bool webserverEnabled_)
{
    // See neural2d.h for more information about these members:

    enableBackPropTraining = true;
    doneErrorThreshold = 0.001f;
    eta = 0.01f;                    // Initial overall net learning rate, [0.0..1.0]
    dynamicEtaAdjust = true;       // true enables automatic eta adjustment during training
    alpha = 0.1f;                   // Momentum factor, multiplier of last deltaWeight, [0.0..1.0]
    lambda = 0.0f;                  // Regularization parameter; disabled if 0.0
    projectRectangular = false;    // Use elliptical areas for sparse connections
    isRunning = true;              // Command line option -p overrides this
    reportEveryNth = 1;
    recentAverageSmoothingFactor = 125.; // Average net errors over this many input samples
    repeatInputSamples = true;
    shuffleInputSamples = true;
    weightsFilename = "weights.txt";
    inputSampleNumber = 0;         // Increments each time feedForward() is called
    error = 1.0f;
    recentAverageError = 1.0f;
    connections.clear();
    layers.clear();
    lastRecentAverageError = 1.0f;
    totalNumberBackConnections = 0;
    totalNumberNeurons = 0;

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    webserverEnabled = webserverEnabled_;
    pLayerToVisualize = nullptr;
    portNumber = 24080;
    if (webserverEnabled) {
        webServer.start(portNumber, messages);
    }
#endif

    // Initialize the dummy bias neuron to provide a weighted bias input for all other neurons.
    // This is a single special neuron that has no inputs of its own, and feeds a constant
    // 1.0 through weighted connections to every other neuron in the network except input
    // neurons:

    bias.output = 1.0f;

    // Set up the layers, create neurons, and connect them:

    if (topologyFilename.size() > 0) {
        parseConfigFile(topologyFilename);  // Throws an exception if any error
    }

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    // Here we construct a string containing items to display in the drop-down
    // visualization menu in the optional GUI interface. The string we construct
    // contains quoted menu options separated by commas suitable for use in
    // Javascript inside a square-bracket array initializer:
    visualizationMenu = "\"None\"";
    for (auto const &pLayer : layers) {
        visualizationMenu.append(pLayer->visualizationsAvailable());
    }
#endif
}


Net::~Net(void) {
#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    if (webserverEnabled) {
        webServer.stopServer();
    }
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

    for (auto &pLayer : layers) {
        pLayer->loadWeights(file);
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

    // The looping order must match that of loadWeights():

    for (auto &pLayer : layers) {
        pLayer->saveWeights(file);
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
    for (auto &n : layers.back()->neurons[0]) { // For all neurons in output layer
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

            for (size_t li = 0; li < layers.back()->neurons[0].size(); ++li) { // Assumes output depth = 1
                auto const &neuron = layers.back()->neurons[0][li];
                if (neuron.output > maxOutput) {
                    maxOutput = neuron.output;
                    maxIdx = li;
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


// Given a layer name and size, create an empty layer. No neurons are created yet.
//
Layer &Net::createLayer(const topologyConfigSpec_t &params)
{
    if (params.isConvolutionFilterLayer) {
        layers.push_back(std::unique_ptr<Layer>(new LayerConvolutionFilter(params)));
    } else if (params.isConvolutionNetworkLayer) {
        layers.push_back(std::unique_ptr<Layer>(new LayerConvolutionNetwork(params)));
    } else if (params.isPoolingLayer) {
        layers.push_back(std::unique_ptr<Layer>(new LayerPooling(params)));
    } else {
        layers.push_back(std::unique_ptr<Layer>(new LayerRegular(params)));
    }

    Layer &newLayer = *layers.back(); // Make a convenient name

    newLayer.resolveTransferFunctionName(params.transferFunctionName);
    newLayer.pConnections = &connections;
    newLayer.projectRectangular = projectRectangular; // Note: cannot be changed after net is initialized. !!!

    return newLayer;
}


// This is an optional way to display lots of information about the network
// topology. Tweak as needed. The argument 'details' can be used to control
// if all the connections are displayed in detail.
//
void Net::debugShowNet(bool details)
{
    info << "\n\nNet configuration (incl. bias connection): --------------------------" << endl;

    for (auto &pLayer : layers) {
        pLayer->debugShow(details);
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

    // Verify that we have the right number of target output values:
    auto const &outputSize = layers.back()->size;
    if (sample.targetVals.size() != outputSize.depth * outputSize.x * outputSize.y) {
        err << "Error: wrong number of target output values in the input data config file" << endl;
        throw exceptionConfigFile();
    }

    // Calculate the gradients of all the neurons' outputs, starting at the output layer:

    for (uint32_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        layers[layerNum]->calcGradients(sample.targetVals);
    }

    // For all layers from outputs to first hidden layer, in reverse order,
    // update connection weights.

    for (uint32_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = *layers[layerNum];
        layer.updateWeights(eta, alpha);
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

    Layer &inputLayer = *layers[0];
    const vector<float> &data = sample.getData(inputLayer.channel);

    if (inputLayer.neurons[0].size() != data.size()) { // We'll assume input layer depth = 1
        err << "Error: input sample " << inputSampleNumber << " has " << data.size()
            << " components, expecting " << inputLayer.neurons[0].size() << endl;
        //throw exceptionRuntime();
    }

    // Rather than make it a fatal error if the number of input neurons != number
    // of input data values, we'll use whatever we can and skip the rest:
    // Assuming the sensible case where the X and Y size of the input layer matches
    // the X and Y size of the input image, then we don't have to flatten the indices
    // because they are already flattened the same way:

    for (uint32_t i = 0; i < (uint32_t)min(inputLayer.neurons[0].size(), data.size()); ++i) {
        inputLayer.neurons[0][i].output = data[i];
    }

    // Start the forward propagation at the first hidden layer:

    std::for_each(layers.begin() + 1, layers.end(), [](std::unique_ptr<Layer> const &pLayer) {
        pLayer->feedForward();
    });

    // If target values are known, update the output neurons' errors and
    // update the overall net error:

    calculateOverallNetError(sample);

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
    // Here is a convenient place to poll for incoming commands from the GUI interface:
    if (webserverEnabled) {
        doCommand();
    }
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

    const Layer &outputLayer = *layers.back();   // A more convenient name

    // Check that the number of target values equals the number of output neurons:
    // Assumes output layer depth = 1

    if (sample.targetVals.size() != outputLayer.neurons[0].size()) {
        err << "Error in sample " << inputSampleNumber << ": wrong number of target values" << endl;
        throw exceptionRuntime();
    }

    for (uint32_t n = 0; n < outputLayer.neurons[0].size(); ++n) {
        float delta = sample.targetVals[n] - outputLayer.neurons[0][n].output;
        error += delta * delta;
    }

    error /= 2.0f * outputLayer.neurons[0].size();

    // Regularization calculations -- this is an experimental implementation.
    // If this experiment works, we should instead calculate the sum of weights
    // on the fly during backprop to see if that is better performance.
    // This adds an error term calculated from the sum of squared weights. This encourages
    // the net to find a solution using small weight values, which can be helpful for
    // multiple reasons.
    // To do: include convolution kernel weights in the sum !!!

    float sumWeightsSquared_ = 0.0;
    if (lambda != 0.0) {
        for (size_t i = 0; i < connections.size(); ++i) {
            sumWeightsSquared_ += connections[i].weight * connections[i].weight;
        }

        error += (sumWeightsSquared_ * lambda)
                     / (2.0f * (totalNumberBackConnections - totalNumberNeurons));
    }

    // Implement a recent average measurement -- average the net errors over N samples:
    lastRecentAverageError = recentAverageError;
    recentAverageError =
            (recentAverageError * recentAverageSmoothingFactor + error)
            / (recentAverageSmoothingFactor + 1.0f);
}


// Returns layer index if found, else returns -1
//
int32_t Net::getLayerNumberFromName(string &name) const
{
    for (auto it = layers.begin(); it != layers.end(); ++it) {
        if ((*it)->layerName == name) {
            return it - layers.begin();
        }
    }

    return -1;
}


// It's possible that some internal neurons don't feed any other neurons.
// That's not a fatal error, but it could be due to an unintentional mistake
// in defining the net topology. Here we will find and report all neurons with
// no forward connections so that the human can fix the topology configuration
// if needed:
void Net::reportUnconnectedNeurons(void)
{
    warn << "\nChecking for neurons with no sinks:" << endl;

    // Loop through all layers except the output layer, looking for unconnected neurons:
    uint32_t neuronsWithNoSink = 0;
    for (uint32_t layerNum = 0; layerNum < layers.size() - 1; ++layerNum) {
        for (auto const &plane : layers[layerNum]->neurons) {
            for (auto const &neuron : plane) {
                if (neuron.forwardConnectionsIndices.size() == 0) {
                    ++neuronsWithNoSink;
                    warn << "  neuron(" << &neuron << ") on " << layers[layerNum]->layerName
                         << endl;
                }
            }
        }
    }
}


// Returns true if the neural net was successfully created and connected. Returns
// false for any error. See the GitHub wiki (https://github.com/davidrmiller/neural2d)
// for more information about the format of the topology config file.
// Throws an exception for any error.
//
void Net::configureNetwork(vector<topologyConfigSpec_t> allLayerSpecs, const string /*configFilename*/)
{
    uint32_t numNeurons = 0;

    // We want to pre-allocate the .layers member so that we can form persistent
    // references to individual layers. We could do this more exactly, but a safe
    // heuristic is to allocate as many layers as elements in the config spec array:
    layers.reserve(allLayerSpecs.size());

    for (topologyConfigSpec_t &spec : allLayerSpecs) {
        // Find indices of existing source and dest layers, or -1 if not found:
        int32_t previouslyDefinedLayerNumSameName = getLayerNumberFromName(spec.layerName);
        int32_t layerNumFrom = getLayerNumberFromName(spec.fromLayerName); // input layer will return -1

        // If the layer of this name does not already exist, create it:
        if (previouslyDefinedLayerNumSameName == -1) {
            info << "Creating layer " << spec.layerName << ", one moment..." << endl;

            // Create a new layer filled with unconnected neurons:
            Layer &newLayer = createLayer(spec);
            newLayer.pConnections = &connections;

            // Pre-allocate all the neurons in the layer so that we can form
            // stable references to individual neurons:
            newLayer.neurons.assign(newLayer.size.depth, vector<Neuron>(newLayer.size.x * newLayer.size.y));
            numNeurons += newLayer.size.x * newLayer.size.y;

            // Connect them:
            if (newLayer.layerName != "input") {
                newLayer.connectLayers(*layers[layerNumFrom]); // Also connect them
            }

            // For some layer types, all neurons get a bias input:
            if (newLayer.layerName != "input" && newLayer.isRegularLayer) {
                newLayer.connectBiasToAllNeuronsAllDepths(bias);
            }

            // For convolution network layers, initialize the kernels and associated data:
            // should the Layer class do this?
            if (spec.isConvolutionNetworkLayer) {
                uint32_t maxNumSourceNeurons = newLayer.size.x * newLayer.size.y;
                for (auto &plane : newLayer.flatConvolveMatrix) {
                    std::for_each(std::begin(plane), std::end(plane), [maxNumSourceNeurons](float &weight) {
                        weight = (float)(((randomFloat() * 2) - 1.0f) / sqrt(maxNumSourceNeurons));
                    });
                }
            }
        } else {
            // Layer already exists, add connections to it.
            // "input" layer will never take this path.
            previouslyDefinedLayerNumSameName = getLayerNumberFromName(spec.layerName);
            Layer &layerTo = *layers[previouslyDefinedLayerNumSameName]; // A more convenient name

            layerTo.connectLayers(*layers[layerNumFrom]);
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
    switch(layers[0]->channel) {
        case NNet::R: channel = "R"; break;
        case NNet::G: channel = "G"; break;
        case NNet::B: channel = "B"; break;
        default:
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

    // Visualization stuff

    // Create the list of visualization menu options:
    s.append("visuals = [ ");
    s.append(visualizationMenu);
    s.append(" ];\r\n");

    if (pLayerToVisualize != nullptr) {
        s.append("selectedVisual = \"" + pLayerToVisualize->layerName + " " + visualizeChoice + "\"\r\n");
    } else {
        s.append("selectedVisual = \"None\"\r\n");
    }

    if (pLayerToVisualize != nullptr) {
        // visualChoice will be either "kernels" or something else. For now, something else always
        // means visualize the layer's outputs. image is a BMP image encoded as a base64 string.
        if (visualizeChoice == "kernels") {
            s.append("image1=\"" + pLayerToVisualize->visualizeKernels() + "\";\r\n");
        } else {
            s.append("image1=\"" + pLayerToVisualize->visualizeOutputs() + "\";\r\n");
        }
    } else {
        s.append("image1=\"\";\r\n"); // Visuals off, empty image.
    }

    return;
}
#endif // end if webserver enabled


#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)
// Handler for the optional external controller for the neural2d program.
// This function reads the command file and acts on any commands received.
//
void Net::actOnMessageReceived(Message_t &msg)
{
    string parameterBlock;
    ColorChannel_t newColorChannel = layers[0]->channel;

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

    // Visualization stuff:
    else if (token.find("vis=") == 0) {
        string visLayer = token.substr(4);
        if (visLayer == "None") {
            pLayerToVisualize = nullptr;
        } else {
            auto itSep = visLayer.find('+');
            assert(itSep != std::string::npos);
            visualizeChoice = visLayer.substr(itSep + 1);
            visLayer = visLayer.substr(0, itSep);
            auto const it = std::find_if(layers.begin(), layers.end(), [&visLayer](std::unique_ptr<Layer> const &pLayer) {
                return pLayer->layerName == visLayer;
            });
            if (it == layers.end()) {
                // If we get here, it indicates a possible mismatch between
                // http-response-template.txt and Net::actOnMessageReceived()
                pLayerToVisualize = nullptr; // Disable visualizations
            } else {
                pLayerToVisualize = it->get();
            }
        }
    }

    // Post processing

    if (newColorChannel != layers[0]->channel) {
        sampleSet.clearImageCache();
        layers[0]->channel = newColorChannel;
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
    if (webserverEnabled) {
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
}
#endif


// Calculate a new eta parameter based on the current and last average net error.
//
float Net::adjustedEta(void)
{
    const float thresholdUp = 0.001f;       // Ignore error increases less than this magnitude
    const float thresholdDown = 0.01f;      // Ignore error decreases less than this magnitude
    const float factorUp = 1.005f;          // Factor to incrementally increase eta
    const float factorDown = 0.999f;        // Factor to incrementally decrease eta

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
