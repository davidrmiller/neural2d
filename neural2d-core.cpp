/*
neural-net.cpp
David R. Miller, 2014
https://github.com/davidrmiller/neural2d

See neural2d.h for more information.

Also see the tutorial video explaining the theory of operation and the
construction of the program.
*/


#include "neural2d.h"
#include <utility>  // for pair()

namespace NNet {


//  ***********************************  Utility functions  ***********************************


// Returns a random float in the range [0.0..1.0]
//
double randomFloat(void)
{
    return (double)rand() / RAND_MAX;
}


// Given an x, y coordinate, return a flattened index.
// There's nothing magic here; we use a function to do this so that we
// always flatten it the same way each time:
//
uint32_t flattenXY(uint32_t x, uint32_t y, uint32_t xSize)
{
    return y * xSize + x;
}


// Assuming an ellipse centered at 0,0 and aligned with the global axes, returns
// a positive value if x,y is outside the ellipse; 0.0 if on the ellipse;
// negative if inside the ellipse.
//
double elliptDist(double x, double y, double radiusX, double radiusY)
{
    assert(radiusX >= 0.0 && radiusY >= 0.0);
    return radiusY*radiusY*x*x + radiusX*radiusX*y*y - radiusX*radiusX*radiusY*radiusY;
}


bool isFileExists(string const &filename)
{
    ifstream file(filename);
    return (bool)file;
}


// Add overloads as needed:
int32_t max(int32_t a, int32_t b) { return a >= b ? a : b; }
int32_t min(int32_t a, int32_t b) { return a <= b ? a : b; }
double absd(double a) { return a < 0.0 ? -a : a; }


// Extracts the X and Y components from the string of the form "XxY".
// E.g., "8x2" will return the pair (8, 2). "8" returns the pair (8, 1).
//
pair<uint32_t, uint32_t> extractTwoNums(const string &s)
{
    uint32_t sizeX;
    uint32_t sizeY;
    char delim;

    istringstream ss(s);

    ss >> sizeX;
    if (ss) {
        ss >> delim;
        if (delim != 'x') {
            sizeY = 1;
        } else {
            ss >> sizeY;
        }
    }

    return pair<uint32_t, uint32_t>(sizeX, sizeY);
}


// ***********************************  Input samples  ***********************************

// Given an image filename and a data container, fill the container with
// data extracted from the image.
//
void ReadBMP(const string &filename, vector<double> &dataContainer)
{
    //assert(expectedNumElements > 0);

    FILE* f = fopen(filename.c_str(), "rb");

    if (f == NULL) {
        cout << "Error reading image file \'" << filename << "\'" << endl;
        // To do: add appropriate error recovery here
        throw "Argument Exception";
    }

    // Read the BMP header to get the image dimensions:

    unsigned char info[54];
    if (fread(info, sizeof(unsigned char), 54, f) != 54) {
        cout << "Error reading the image header from \'" << filename << "\'" << endl;
        assert(false);
    }

    if (info[0] != 'B' || info[1] != 'M') {
        cout << "Error: invalid BMP file \'" << filename << "\'" << endl;
        throw("Invalid BMP file");
    }

    // Verify the offset to the pixel data. It should be the same size as the info[] data read above.

    size_t dataOffset = (info[13] << 24)
                      + (info[12] << 16)
                      + (info[11] << 8)
                      +  info[10];

    // Verify that the file contains 24 bits (3 bytes) per pixel (red, green blue at 8 bits each):

    int pixelDepth = (info[29] << 8) + info[28];
    if (pixelDepth != 24) {
        cout << "Error: BMP file is not 24 bits per pixel" << endl;
        throw("Unsupported image file format");
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
        //cout << "Error seeking to offset " << dataOffset << " in file \'" << filename << "\'" << endl;
        throw("Error seeking in BMP file");
    }

    uint32_t rowLen_padded = (width*3 + 3) & (~3);
    unsigned char *imageData = new unsigned char[rowLen_padded];

    dataContainer.clear();

    // Fill the data container with 8-bit data taken from the image data:

    for (uint32_t y = 0; y < height; ++y) {
        if (fread(imageData, sizeof(unsigned char), rowLen_padded, f) != rowLen_padded) {
            cout << "Error reading \'" << filename << "\' row " << y << endl;
            // To do: add appropriate error recovery here
            throw("Error reading image file");
        }

        // Bytes are arranged in the order (B, G, R). Enable one of the conversions
        // below to convert the RGB pixel to a single unsigned char:

        for (uint32_t x = 0; x < width; ++x) {
            //pPixelsBW[x * width + y] = imageData[x * 3 + 0];  // use the blue pixel value
            //pPixelsBW[x * width + y] = imageData[x * 3 + 1];  // use the green pixel value
            //pPixelsBW[x * height + y] = imageData[x * 3 + 2]; // use the red pixel value
//            pPixelsBW[x * height + y] =  // convert RBG to BW
//                                         0.3 * imageData[x*3 + 2] +   // red
//                                         0.6 * imageData[x*3 + 1] +   // green
//                                         0.1 * imageData[x*3 + 0];    // blue
            unsigned char val =
                                 0.3 * imageData[x*3 + 2] +   // red
                                 0.6 * imageData[x*3 + 1] +   // green
                                 0.1 * imageData[x*3 + 0];    // blue

            // Convert the 8-bit value to a double while caching it:
            dataContainer.push_back(val / 256.0);
        }
    }

    fclose(f);
    delete[] imageData;
}

// If this is the first time getData() is called, we'll open the image file and
// cache the pixel data in memory.
// Returns a reference to the container of input data.
//
vector<double> &Sample::getData(void)
{
   if (data.size() == 0) {
       ReadBMP(imageFilename, data);
   }

   // If we get here, we can assume there is something in the .data member

   return data;
}


// Given the name of an input sample config file, open it and save the contents
// in memory. For now, we'll just read the image filenames and, if available, the
// target output values. We'll defer reading the image pixel data until it's needed.
//
SampleSet::SampleSet(const string &inputFilename)
{
    string line;
    uint32_t lineNum = 0;

    if (!isFileExists(inputFilename)) {
        cout << "Error reading input samples config file \'" << inputFilename << "\'" << endl;
        throw("Error reading input samples config file");
    }

    ifstream dataIn(inputFilename);
    if (!dataIn || !dataIn.is_open()) {
        cout << "Error opening input samples config file \'" << inputFilename << "\'" << endl;
        throw("Error reading input samples config file");
    }

    samples.clear();  // Lose all prior samples

    while (getline(dataIn, line)) {
        ++lineNum;
        Sample sample; // Default ctor will clear all members

        stringstream ss(line);
        ss >> sample.imageFilename;
        // Skip blank and comment lines:
        if (sample.imageFilename.size() == 0 || sample.imageFilename[0] == '#') {
            continue;
        }

        // If they exist, read the target values from the rest of the line:
        while (!ss.eof()) {
            double val;
            if (!(ss >> val).fail()) {
                sample.targetVals.push_back(val);
            }
        }

        samples.push_back(sample);
    }

    cout << samples.size() << " training samples initialized" << endl;
}


// Randomize the order of the samples container.
//
void SampleSet::shuffle(void)
{
    random_shuffle(samples.begin(), samples.end());
}


// ***********************************  struct Connection  ***********************************


Connection::Connection(Neuron &from, Neuron &to)
     : fromNeuron(from), toNeuron(to)
{
}


// ***********************************  Transfer Functions  ***********************************

// Here is where we define at least one transfer function to use. We refer to them by
// name, where "" is an alias for the default function. To select a different one,
// add a "tf" parameter to the layer definition in the topology config file. All the
// neurons in any one layer will use the same transfer function.

// tanh is a sigmoid curve scaled; output ranges from -1 to +1:
double transferFunctionTanh(double x) { return tanh(x); }
double transferFunctionDerivativeTanh(double x) { return 1.0 - tanh(x) * tanh(x); }

// logistic is a sigmoid curve that ranges 0.0 to 1.0:
double transferFunctionLogistic(double x) { return 1.0 / (1.0 + exp(-x)); }
double transferFunctionDerivativeLogistic(double x) { return x * (1.0 - x); }

// linear is a constant slope; ranges from -inf to +inf:
double transferFunctionLinear(double x) { return x; }
double transferFunctionDerivativeLinear(double x) { return (void)x, 1.0; }

// ramp is a constant slope between -1 <= x <= 1, zero slope elsewhere; output ranges from -1 to +1:
double transferFunctionRamp(double x)
{
    if (x < -1.0) return -1.0;
    else if (x > 1.0) return 1.0;
    else return x;
}
double transferFunctionDerivativeRamp(double x) { return (x < -1.0 || x > 1.0) ? 0.0 : 1.0; }

// gaussian:
double transferFunctionGaussian(double x) { return exp(pow(-x, 2.0)); }
double transferFunctionDerivativeGaussian(double x) { return -2.0 * x * exp(pow(-x, 2.0)); }



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
Neuron::Neuron(vector<Connection> *pConnectionsData, const string &transferFunctionName)
{
    output = randomFloat() - 0.5;
    gradient = 0.0;
    pConnections = pConnectionsData; // Remember where the connections array is
    backConnectionsIndices.clear();
    forwardConnectionsIndices.clear();
    sourceNeurons.clear();

    if (transferFunctionName == "" || transferFunctionName == "tanh") {
        // This is the default transfer function:
        transferFunction = transferFunctionTanh;
        transferFunctionDerivative = transferFunctionDerivativeTanh;
    } else if (transferFunctionName == "logistic") {
        transferFunction = transferFunctionLogistic;
        transferFunctionDerivative = transferFunctionDerivativeLogistic;
    } else if (transferFunctionName == "linear") {
        transferFunction = transferFunctionLinear;
        transferFunctionDerivative = transferFunctionDerivativeLinear;
    } else if (transferFunctionName == "ramp") {
        transferFunction = transferFunctionRamp;
        transferFunctionDerivative = transferFunctionDerivativeRamp;
    } else if (transferFunctionName == "gaussian") {
        transferFunction = transferFunctionGaussian;
        transferFunctionDerivative = transferFunctionDerivativeGaussian;
    } else {
        cout << "Undefined transfer function: \'" << transferFunctionName << "\'" << endl;
    }
}


// The error gradient of an output-layer neuron is equal to the target (desired)
// value minus the computed output value, times the derivative of
// the output-layer activation function evaluated at the computed output value.
//
void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - output;
    gradient = delta * Neuron::transferFunctionDerivative(output);
}


// The error gradient of a hidden-layer neuron is equal to the derivative
// of the activation function of the hidden layer evaluated at the
// local output of the neuron times the sum of the product of
// the primary outputs times their associated hidden-to-output weights.
//
void Neuron::calcHiddenGradients(void)
{
    double dow = sumDOW_nextLayer();
    gradient = dow * Neuron::transferFunctionDerivative(output);
}


// To do: add commentary!!!
//
double Neuron::sumDOW_nextLayer(void) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (auto idx : forwardConnectionsIndices) {
        const Connection &conn = (*pConnections)[idx];

        //sum += conn.weight * (conn.pToNeuron)->gradient;
        sum += conn.weight * conn.toNeuron.gradient;
    }

    return sum;
}


void Neuron::updateInputWeights(double eta, double alpha)
{
    // The weights to be updated are the weights from the neurons in the
    // preceding layer (the source layer) to this neuron:

    for (auto idx : backConnectionsIndices) {
        Connection &conn = (*pConnections)[idx];
        //const Neuron &neuron = *(conn.pFromNeuron);

        const Neuron &fromNeuron = conn.fromNeuron;
        double oldDeltaWeight = conn.deltaWeight;

        double newDeltaWeight =
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
    double sum = 0.0;

    // Sum the neuron's inputs:
    for (auto idx : this->backConnectionsIndices) {
        const Connection &conn = (*pConnections)[idx];

        sum += conn.fromNeuron.output * conn.weight;
    }

    // Shape the output by passing it through the transfer function:
    this->output = Neuron::transferFunction(sum);
}


// ***********************************  class Net  ***********************************


Net::Net(const string &topologyFilename)
{
    // See nnet.h for descriptions of these variables:

    eta = 0.01;               // Initial overall net learning rate, [0.0..1.0]
    dynamicEtaAdjust = true;  // true enables automatic eta adjustment during training
    alpha = 0.1;              // Momentum factor, multiplier of last deltaWeight, [0.0..1.0]
    lambda = 0.0;             // Regularization parameter; disabled if 0.0
    enableRemoteInterface = true;  // if true, causes the net to respond to remote commands
    isRunning = !enableRemoteInterface;
    totalNumberConnections = 0;
    totalNumberNeurons = 0;
    sumWeights = 0.0;
    repeatInputSamples = true;
    inputSampleNumber = 0;    // Increments each time feedForward() is called

    // Init the optional remote command interface: For now, we'll establish the
    // filename for the command file, but set the file descriptor to an invalid
    // value to show that it is not yet open. The function doCommand() will
    // open the file when it is needed.

#ifndef _WIN32
    cmdFd = -1;
#endif
    if (enableRemoteInterface) {
#ifdef _WIN32
        cmdFilename = "neural2d-command";
#else
        cmdFilename = "neural2d-command-" + to_string((unsigned)getppid());
#endif
    } else {
        cout << "Remote command interface disabled." << endl;
        isRunning = true;   // Start immediately, don't wait for a remote command
    }
    
    // The recent average smoothing factor helps us to better see how training is
    // proceeding by averaging the overall network errors over a number of recent
    // training samples:

    recentAverageSmoothingFactor = 100.;
    lastRecentAverageError = recentAverageError = 1.0;

    // Initialize the dummy bias neuron to provide a weighted bias input for all other neurons.
    // This is a single special neuron that has no inputs of its own, and feeds a constant
    // 1.0 through weighted connections to every other neuron in the network except input
    // neurons:

    bias.output = 1.0;
    bias.gradient = 0.0;
    bias.pConnections = &connections;

    // Set up the layers, create neurons, and connect them:

    parseConfigFile(topologyFilename);  // Throws an exception if any error
}


// Load weights from an external file. The file must contain one floating point
// number per line, with no blank lines. This function is intended to read the
// same format that saveWeights() produces.
//
bool Net::loadWeights(const string &filename)
{
    if (!isFileExists(filename)) {
        cout << "Error reading weights file \'" << filename << "\'" << endl;
        throw("Error reading weights file");
    }

    ifstream file(filename);
    if (!file) {
        throw "Invalid weights file";
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
    ofstream file(filename);
    if (!file) {
        throw "Error opening weights file";
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

    cout << "\nPass #" << inputSampleNumber << ": " << sample.imageFilename << "\nOutputs: ";
    for (auto &n : layers.back().neurons) { // For all neurons in output layer
        cout << n.output << " ";
    }
    cout << endl;

    if (sample.targetVals.size() > 0) {
        cout << "Expected ";
        for (double targetVal : sample.targetVals) {
            cout << targetVal << " ";
        }
    }

    // Optional: Enable the following block if you would like to report the net's
    // outputs and the expected values as Booleans, where any value <= 0 is considered
    // false, and > 0 is considered true. This can be used, e.g., for pattern
    // recognition where each output neuron corresponds to one pattern class,
    // and the output neurons are trained to be positive or negative to indicate
    // true or false.

    if (true) {
        //double maxOutput = (numeric_limits<double>::min)();
        double maxOutput = -1.e8;
        size_t maxIdx = 0;

        for (size_t i = 0; i < layers.back().neurons.size(); ++i) {
            Neuron const &n = layers.back().neurons[i];
            if (n.output > maxOutput) {
                maxOutput = n.output;
                maxIdx = i;
            }
        }

        if (sample.targetVals[maxIdx] > 0.0) {
            cout << " " << string("Correct");
        } else {
            cout << " " << string("Wrong");
        }
        cout << endl;
    }

    // Optionally enable the following line to display the current eta value
    // (in case we're dynamically adjusting it):
    cout << "  eta=" << eta;

    // Show overall net error for this sample and for the last few samples averaged:
    cout << "Net error = " << error << ", running average = " << recentAverageError << endl;


}


// Given an existing layer with neurons already connected, add more
// connections. This happens when a layer specification is repeated in
// the config file, thus creating connections to source neurons from
// multiple layers. Returns false for any error, true if successful.
//
bool Net::addToLayer(Layer &layerTo, Layer &layerFrom,
        uint32_t sizeX, uint32_t sizeY, uint32_t radiusX, uint32_t radiusY)
{
    if (sizeX != layerTo.sizeX || sizeY != layerTo.sizeY) {
        cerr << "Error: Config: repeated layer '" << layerTo.name
             << "' must be the same size" << endl;
        return false;
    }

    // Layer already exists -- add more backConnections to the existing layer:

    for (uint32_t ny = 0; ny < layerTo.sizeY; ++ny) {
        cout << "\r" << ny << flush; // Progress indicator

        for (uint32_t nx = 0; nx < layerTo.sizeX; ++nx) {
            //cout << "connect to neuron " << nx << "," << ny << endl;
            connectNeuron(layerTo, layerFrom, layerTo.neurons[flattenXY(nx, ny, layerTo.sizeX)],
                          nx, ny, radiusX, radiusY);
            // n.b. Bias connections were already made when the neurons were first created.
        }
    }

    cout << endl; // End progress indicator

    return true;
}


// Given a layer name and size, create an empty layer. No neurons are created yet.
//
Layer &Net::createLayer(const string &name, uint32_t sizeX, uint32_t sizeY)
{
    layers.push_back(Layer());
    Layer &layer = layers.back(); // Make a convenient name

    layer.name = name;
    layer.sizeX = sizeX;
    layer.sizeY = sizeY;

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

    cout << "\n\nNet configuration (not including bias connections): --------------------------" << endl;
    for (auto const &l : layers) {
        numFwdConnections = 0;
        numBackConnections = 0;
        cout << "Layer '" << l.name << "' has " << l.neurons.size()
             << " neurons arranged in " << l.sizeX << "x" << l.sizeY << endl;
        for (auto const &n : l.neurons) {
            numFwdConnections += n.forwardConnectionsIndices.size();
            numBackConnections += n.backConnectionsIndices.size(); // Includes the bias connection
            if (l.name != "input") {
                --numBackConnections; // Ignore the bias connection
            }
            if (details && n.forwardConnectionsIndices.size() > 0) {
                cout << "  neuron(" << &n << ")" << endl;
                if (n.backConnectionsIndices.size() > 0) { // Includes the bias connection
                    cout << "    Back connections (incl. bias):" << endl;
                    for (auto idx : n.backConnectionsIndices) {
                        Connection const &c = connections[idx];
                        cout << "      conn(" << &c << ") pFrom=" << &c.fromNeuron
                             << ", pto=" << &c.toNeuron << endl;
                        assert(&c.toNeuron == &n);
                    }
                }

                if (n.forwardConnectionsIndices.size() > 0) { // Redundant
                    cout << "    Fwd connections:" << endl;
                    for (auto idx : n.forwardConnectionsIndices) {
                        Connection const &pc = connections[idx];
                        cout << "      conn->pFrom=" << &pc.fromNeuron
                             << ", ->pTo=" << &pc.toNeuron << endl;
                    }
                }
            }
        }
        if (!details) {
            cout << "   connections: " << numBackConnections << " back, "
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
    // update connection weights:

    // Optionally enable the following #pragma line to permit OpenMP to
    // parallelize this loop. For gcc 4.x, add the option "-fopenmp" to
    // the compiler command line. If the compiler does not understand
    // OpenMP, the #pragma will be ignored.
#pragma omp parallel for

    for (uint32_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = layers[layerNum];

        for (auto &neuron : layer.neurons) {
            neuron.updateInputWeights(eta, alpha);
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

    const vector<double> &data = sample.getData();
    Layer &inputLayer = layers[0];

    if (inputLayer.neurons.size() != data.size()) {
        cout << "Error: input sample " << inputSampleNumber << " has " << data.size()
             << " components, expecting " << inputLayer.neurons.size() << endl;
        throw("Wrong number of inputs");
    }

    for (uint32_t i = 0; i < data.size(); ++i) {
        inputLayer.neurons[i].output = data[i];
    }

    // Start the forward propagation at the first hidden layer:

    for (uint32_t layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
        Layer &layer = layers[layerIdx];

        // Optionally enable the following #pragma line to permit OpenMP to
        // parallelize this loop. For gcc 4.x, add the option "-fopenmp" to
        // the compiler command line. If the compiler does not understand
        // OpenMP, the #pragma will be ignored. This is shown as an example.
        // There are probably other loops in the program that could be
        // parallelized to get even better performance.

#pragma omp parallel for

        for (uint32_t i = 0; i < layer.neurons.size(); ++i) {
            layer.neurons[i].feedForward();
        }
    }

    // If target values are known, update the output neurons' errors and
    // update the overall net error:

    calculateOverallNetError(sample);

    // Here is a convenient place to poll for incoming commands from the GUI interface:

    doCommand();
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

    Layer &outputLayer = layers.back();

    // Check that the number of target values equals the number of output neurons:
    if (sample.targetVals.size() != outputLayer.neurons.size()) {
        cout << "Error in sample " << inputSampleNumber << ": wrong number of target values" << endl;
        throw("Wrong number of target values");
    }

    for (uint32_t n = 0; n < outputLayer.neurons.size(); ++n) {
        double delta = sample.targetVals[n] - outputLayer.neurons[n].output;
        error += delta * delta;
    }

    error /= 2.0 * outputLayer.neurons.size();

    // Regularization calculations -- if this works, calculate the sum of weights on the fly
    // during backprop:

    double sumWeightsSquared = 0.0;
    if (lambda != 0.0) {
        // For all layers except the input layer, sum all the weights. These are in
        // the back-connection records:
        for (uint32_t layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
            Layer &layer = layers[layerIdx];
            for (auto const &neuron : layer.neurons) {
                // To do: skip the bias connection correctly; it's usually, but not always,
                // the last connection in the list
                for (uint32_t idx = 0; idx < neuron.backConnectionsIndices.size() - 1; ++idx) {
                    Connection &conn = (*neuron.pConnections)[neuron.backConnectionsIndices[idx]];
                    sumWeightsSquared += conn.weight * conn.weight;
                }
            }
        }

        double sumWeights = (sumWeightsSquared * lambda)
                          / (2.0 * (totalNumberConnections - totalNumberNeurons));
        error += sumWeights;
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
// The location of the destination neuron is projected onto the neurons of the source
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
        uint32_t nx, uint32_t ny, uint32_t radiusX, uint32_t radiusY)
{
    bool projectRectangular = false;  // If false, do an elliptical pattern

    uint32_t sizeX = layerTo.sizeX;
    uint32_t sizeY = layerTo.sizeY;
    assert(sizeX > 0 && sizeY > 0);

    // Calculate the normalized [0..1] coordinates of our neuron:
    float normalizedX = ((float)nx / sizeX) + (1.0 / (2 * sizeX));
    float normalizedY = ((float)ny / sizeY) + (1.0 / (2 * sizeY));

    // Calculate the coords of the nearest neuron in the "from" layer.
    // The calculated coords are relative to the "from" layer:
    uint32_t lfromX = uint32_t(normalizedX * fromLayer.sizeX); // should we round off instead of round down?
    uint32_t lfromY = uint32_t(normalizedY * fromLayer.sizeY);

    //cout << "our neuron at " << nx << "," << ny << " covers neuron at " 
    //     << lfromX << "," << lfromY << endl;

    // Calculate the rectangular window into the "from" layer:

    uint32_t xmin = (uint32_t)max(lfromX - radiusX, 0);
    uint32_t xmax = (uint32_t)min(lfromX + radiusX, fromLayer.sizeX - 1);

    uint32_t ymin = (uint32_t)max(lfromY - radiusY, 0);
    uint32_t ymax = (uint32_t)min(lfromY + radiusY, fromLayer.sizeY - 1);

    // Now (xmin,xmax,ymin,ymax) defines a rectangular subset of neurons in a previous layer.
    // We'll make a connection from each of those neurons in the previous layer to our
    // neuron in the current layer.

    // We will also check for and avoid duplicate connections. Duplicates are mostly harmless,
    // but unnecessary. Duplicate connections can be formed when the same layer name appears
    // more than once in the topology config file with the same "from" layer if the projected
    // rectangular or elliptical areas on the source layer overlap.

    double xcenter = ((double)xmin + (double)xmax) / 2.0;
    double ycenter = ((double)ymin + (double)ymax) / 2.0;
    uint32_t maxNumSourceNeurons = ((xmax - xmin) + 1) * ((ymax - ymin) + 1);

    for (uint32_t y = ymin; y <= ymax; ++y) {
        for (uint32_t x = xmin; x <= xmax; ++x) {
            if (!projectRectangular && elliptDist(xcenter - x, ycenter - y, radiusX, radiusY) > 1.0) {
                continue; // Skip this location, it's outside the ellipse
            }
            Neuron &fromNeuron = fromLayer.neurons[flattenXY(x, y, fromLayer.sizeX)];

            bool duplicate = false;
            if (neuron.sourceNeurons.find(&fromNeuron) != neuron.sourceNeurons.end()) {
                cout << "dup" << endl;
                duplicate = true;
                break; // Skip this connection, proceed to the next
            }

            if (!duplicate) {
                // Add a new Connection record to the main container of connections:
                connections.push_back(Connection(fromNeuron, neuron));
                int connectionIdx = connections.size() - 1;  //    and get its index,
                //Connection &conn = connections.back();       //    and a convenient reference
                ++totalNumberConnections;

                // Initialize the weight of the connection:
                connections.back().weight = (randomFloat() - 0.5) / maxNumSourceNeurons;

                // Record the back connection index at the destination neuron:
                neuron.backConnectionsIndices.push_back(connectionIdx);

                // Remember the source neuron for detecting duplicate connections:
                neuron.sourceNeurons.insert(&fromNeuron);

                //conn.pFromNeuron = &fromNeuron;
                //conn.pToNeuron = &neuron;
                //uint32_t flatIdxTo = nx * sizeY + ny;
                //assert(&neuron == &layerTo.neurons[flatIdxTo]); // Just checking

                // Record the Connection index at the source neuron:
                uint32_t flatIdxFrom = flattenXY(x, y, fromLayer.sizeX); // x * fromLayer.sizeY + y;
                fromLayer.neurons[flatIdxFrom].forwardConnectionsIndices.push_back(
                           connectionIdx);

//                cout << "    connect from layer " << fromLayer.name
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
    c.deltaWeight = 0.0;

    //c.pToNeuron = &neuron;
    //c.pFromNeuron = &bias; // Point to the fake neuron with constant 1.0 output

    // Record the back connection with the destination neuron:
    //neuron.backConnectionsIndices.push_back(connectionIdx);
    //neuron.backConnections.push_back()
    neuron.backConnectionsIndices.push_back(connectionIdx);

    ++totalNumberConnections;

    // Record the forward connection with the fake bias neuron:
    bias.forwardConnectionsIndices.push_back(connectionIdx);
}


// Returns layer index if found, else returns -1
//
int32_t Net::getLayerNumberFromName(const string &name) const
{
    for (uint32_t ln = 0; ln < layers.size(); ++ln) {
        if (layers[ln].name == name) {
            return (int32_t)ln;
        }
    }

    return -1;
}


// Create neurons and connect them. For the input layer, there are no incoming
// connections and radius doesn't apply. Calling this function with layerFrom == layerTo
// indicates an input layer.
//
void Net::createNeurons(Layer &layerTo, Layer &layerFrom, const string &transferFunctionName,
                        uint32_t radiusX, uint32_t radiusY)
{
    // Reserve enough space in layer.neurons to prevent reallocation (so that
    // we can form stable references to neurons):

    layerTo.neurons.reserve(layerTo.sizeX * layerTo.sizeY);

    for (uint32_t ny = 0; ny < layerTo.sizeY; ++ny) {

        cout << "\r" << ny << "/" << layerTo.sizeY << flush; // Progress indicator

        for (uint32_t nx = 0; nx < layerTo.sizeX; ++nx) {
            // When we create a neuron, we have to give it a pointer to the
            // start of the array of Connection objects:
            layerTo.neurons.push_back(Neuron(&connections, transferFunctionName));
            Neuron &neuron = layerTo.neurons.back(); // Make a more convenient name
            ++totalNumberNeurons;

//          cout << "  created neuron(" << &layerTo.neurons.back() << ")"
//               << " at " << nx << "," << ny << endl;

            // If layerFrom is layerTo, it means we're making input neurons
            // that have no input connections to the neurons. Else, we must make connections
            // to the source neurons and to a bias input:

            if (&layerFrom != &layerTo) {
                connectNeuron(layerTo, layerFrom, neuron,
                              nx, ny, radiusX, radiusY);
                connectBias(neuron);
            }
        }
    }

    cout << endl; // End the progress indicator
}


/*
   Returns true if the neural net was successfully created and connected. Returns
   false for any error.

   Config file notes:

   There must be a layer named "input" specified first.
   There must be a layer named "output specified last.
   Hidden layers can be named anything starting with "layer".
   The size parameter refers to the number of neurons in the destination layer (layer to the right).
   The radius parameter refers to the number of neurons in the source layer (layer to the left).
   A radius of 0 means: connect from just one source input neuron.
   A radius of 1 means: connect from the corresponding source neuron and its immediate neighbors.
   A radius larger than the previous layer (or an unspecified radius) harmlessly encompasses all
   the neurons in the layer.
   A layer name may be repeated to connect with source neurons in multiple source layers.
   A repeated layer name must specify identical size parameters.

   Example config file:

   # Comment lines start with #
   # Blank lines are ignored.
   input size 64x64
   layerV from input size 32x32 radius 1x32 tf 0
   layerH from input size 32x32 radius 32x1
   layerVH from layerV size 16x16 radius 32x32
   layerVH from layerH size 16x16 radius 32x32     <-- repeated layer must have same size
   layerC from layerVH size 8x8                    <-- no radius = fully connected
   output from layerC size 2x1 radius 8x8 tf 1
*/

// Throws an exception for any error.
//
void Net::parseConfigFile(const string &configFilename)
{
    if (!isFileExists(configFilename)) {
        cout << "Error reading topology file \'" << configFilename << "\'" << endl;
        throw("Error reading topology config file");
    }

    ifstream cfg(configFilename);
    if (!cfg) {
        throw "Error opening topology config file";
    }

    // Reserve enough space in the .layers container to prevent reallocation.
    // Heuristic: count non-blank and non-comment lines in the config file
    // to get an upper limit of the number of layers and use that as the max
    // capacity to reserve. It might reserve more space than actually needed, but
    // normally that would be insignificant.

    unsigned lineNum = 0;
    string line;
    while (getline(cfg, line)) {
        if (line[0] != '\n' && line[0] != '\0' && line[0] != '#') {
            ++lineNum;
        }
    }
    layers.reserve(lineNum);

    // Reset the stream pointer and parse the config file for real now:

    int32_t previouslyDefinedLayerNumSameName;
    uint32_t numNeurons = 0;
    lineNum = 0;
    cfg.clear();   // Reset the stream EOF
    cfg.seekg(0);  // Rewind the stream, start over

    while (getline(cfg, line)) {
        ++lineNum;
        istringstream ss(line);

        // Here are the parameters that we will try to extract from each config line:

        string layerName = "";             // Can be input, output, or layer*
        string fromLayerName = "";         // Can be any existing layer name
        uint32_t sizeX = 0;                // Format: size XxY
        uint32_t sizeY = 0;
        uint32_t radiusX = 1e9;            // Format: radius XxY
        uint32_t radiusY = 1e9;
        string transferFunctionName = "";  // Format: tf name

        bool done = false;
        while (!done && !ss.eof()) {
            string token;
            char delim;
            ss >> layerName;  // First field is the layer name

            // Skip blank and comment lines:
            if (layerName == "" || layerName[0] == '#') {
                done = true;
                continue;
            }

            // There will be one or more additional fields:

            while (!ss.eof()) {
                pair<uint32_t, uint32_t> twoNums;
                string tempString;

                ss >> token;
                if (token == "from") {
                    ss >> fromLayerName;
                } else if (token == "size") {
                    //ss >> sizeX >> delim >> sizeY;  // Size XxY
                    ss >> tempString;
                    twoNums = extractTwoNums(tempString);
                    sizeX = twoNums.first;
                    sizeY = twoNums.second;
                } else if (token == "radius") {
                    ss >> radiusX >> delim >> radiusY;
                } else if (token == "tf") {
                    ss >> transferFunctionName;
                } else {
                    cout << "Syntax error in " << configFilename << " line " << lineNum << endl;
                    throw("Topology config file syntax error");
                }
            }

            // Now do something with the data we extracted:

            // If this is the first layer, check that it is called "input":

            if (layers.size() == 0 && layerName != "input") {
                cerr << "Error in " << configFilename << ": the first layer must be named 'input'" << endl;
                throw("Topology config file syntax error");
            }

            int32_t layerNumFrom = getLayerNumberFromName(fromLayerName); // input layer will return -1

            if (layerName != "input") {
                // If not input layer, then the source layer must already exist:
                if (layerNumFrom == -1) {
                    cerr << "Error: Config(" << lineNum << "): from layer '"
                         << fromLayerName << "' is undefined." << endl;
                    throw("Topology config file syntax error");
                }
            } else {
                // For input layer, from layer is not allowed:
                if (fromLayerName != "") {
                    cerr << "Error: Config(" << lineNum << "): "
                         << " input layer cannot have a 'from' parameter" << endl;
                    throw("Topology config file syntax error");
                }
            }

            // Check if a layer already exists with the same name:

            previouslyDefinedLayerNumSameName = getLayerNumberFromName(layerName);
            if (previouslyDefinedLayerNumSameName == -1) {

                // To do: Add range check for sizeX, sizeY, radiusX, radiusY params !!!

                // Create a new layer of this name.
                // "input" layer will always take this path.

                Layer &newLayer = createLayer(layerName, sizeX, sizeY);

                // Create neurons and connect them:

                cout << "Creating layer " << layerName << ", one moment..." << endl;
                if (layerName == "input") {
                    createNeurons(newLayer, newLayer, transferFunctionName); // Input layer has no back connections
                } else {
                    createNeurons(newLayer, layers[layerNumFrom], transferFunctionName,
                                  radiusX, radiusY); // Also connects them
                }
                numNeurons += sizeX * sizeY;
            } else {

                // Layer already exists, add connections to it.
                // "input" layer will never take this path.

                Layer &layerToRef = layers[previouslyDefinedLayerNumSameName]; // A more convenient name
                if (layerToRef.sizeX != sizeX || layerToRef.sizeY != sizeY) {
                    cerr << "Error: Config(" << lineNum << "): layer '" << layerName
                         << "' already exists, but different size" << endl;
                    throw("Topology config file syntax error");
                }

                // Add more connections to the existing neurons in this layer:

                bool ok = addToLayer(layerToRef,
                                     layers[layerNumFrom],
                                     sizeX, sizeY, radiusX, radiusY);
                if (!ok) {
                    cout << "Error in " << configFilename << ", layer \'" << layerName << "\'" << endl;
                    throw("Error adding connections");
                }
            }
        }
    }

    // Check that the last layer defined is named "output":

    if (layers.back().name != "output") {
        cerr << "Error in " << configFilename << ": the last layer must be named 'output'" << endl;
        throw("Topology config file syntax error");
    }

    // It's possible that some internal neurons don't feed any other neurons.
    // That's not a fatal error, but it's probably due to an unintentional mistake
    // in defining the net topology. Here we will find and report all neurons with
    // no forward connections so that the human can fix the topology configuration
    // if needed:

    cout << "\nChecking for neurons with no sinks:" << endl;

    // Loop through all layers except the output layer, looking for unconnected neurons:
    uint32_t neuronsWithNoSink = 0;
    for (uint32_t lidx = 0; lidx < layers.size() - 1; ++lidx) {
        Layer const &layer = layers[lidx];
        for (auto const &neuron : layer.neurons) {
            if (neuron.forwardConnectionsIndices.size() == 0) {
                ++neuronsWithNoSink;
                cout << "  neuron(" << &neuron << ") on " << layer.name
                     << endl;
            }
        }
    }

    // Optionally enable the next line to display the resulting net topology:
    debugShowNet();

    cout << "\nConfig file parsed successfully." << endl;
    cout << "Found " << neuronsWithNoSink << " neurons with no sink." << endl;
    cout << numNeurons << " neurons total; " << totalNumberConnections
         << " back+bias connections." << endl;
    cout << "About " << (int)((float)totalNumberConnections / numNeurons + 0.5)
         << " connections per neuron on average." << endl;
}


// Handler for the optional external controller for the neural2d program.
// This function reads the command file and acts on any commands received.
// The command file is named cmdFilename, which by default is
// "neural2d-command-" plus the process ID of the parent process.
// Returns when all outstanding commands have been executed. Returns
// immediately if no commands have arrived, or if the remote command interface
// is disabled.
//
// Commands:
//     alpha n               set momentum, float n > 0.0
//     data filename         init training samples
//     dynamiceta True|False enable/disable dynamic updating of the eta parameter
//     eta n                 set learning rate, float n > 0.0
//     load filename         load weights from a file
//     pause                 pause operation
//     report n              progress report interval, int n >= 1
//     resume                resume operation (alias for "run")
//     run                   resume operation (alias for "resume")
//     save filename         save weights (but not topology)
//     train True|False      enable/disable backprop weight updates
//
void Net::doCommand()
{
    if (!enableRemoteInterface) {
        return;
    }

    // Typically fewer than 10 or so commands may have been written to the command file by
    // the controlling program before we have the chance to read and act on them.
    // The size of buf[] should be large enough to hold the outstanding command strings.

    char buf[1024];

    // Check if the optional interface is enabled and active. If it's enabled but not
    // yet open, we'll open it and leave it open. We can tell that the file is already
    // open if cmdFd is nonnegative. We must open it in non-blocking mode so that we
    // don't hang in read() if no commands have arrived.

#ifdef _WIN32
    if (!cmdStream.is_open() && isFileExists(cmdFilename)) {
        cout << "opening command interface file \'" << cmdFilename << "\'" << endl;
        cmdStream.open(cmdFilename);
        if (!cmdStream.is_open()) {
            // The command interface file exists but we can't open it
            cout << "Error opening remote command interface file \'"
                 << cmdFilename << "\'" << endl;
            throw("Error opening command interface file");
        }
        
#else
    if (cmdFd < 0 && isFileExists(cmdFilename)) {
        cout << "opening command interface file \'" << cmdFilename << "\'" << endl;
        cmdFd = open(cmdFilename.c_str(), O_RDONLY | O_NONBLOCK);
        if (cmdFd < 0) {
            // The command interface file exists but we can't open it
            cout << "Error opening remote command interface file \'"
                 << cmdFilename << "\'" << endl;
            throw("Error opening command interface file");
        }
#endif
    } else if (!isFileExists(cmdFilename)) {
        // If we get here, it's because enableRemoteInterface is true, but the command file
        // does not exist. Instead of waiting forever for a command that will never arrive,
        // we'll instead set isRunning to true so that the net can start processing data.
        isRunning = true;
    }

    do {
        // If the command file is open, see if there's anything there to read. If not,
        // return immediately.

        int numRead = 0;
    #ifdef _WIN32
        if (cmdStream.is_open() && cmdStream.good()) {
            cmdStream.read(buf, sizeof buf - 1);
            numRead = cmdStream.gcount();
    #else
        if (cmdFd >= 0) {
            numRead = read(cmdFd, buf, sizeof buf - 1);
    #endif
            if (numRead == 0 && isRunning) {
                // eof
                return;
            } else if (numRead < 0) {
                // errno 11 is EAGAIN (try again) which may be the same on Linux as EWOULDBLOCK
                if (errno != EAGAIN) {
                    cout << "Error in command-and-control stream, errno=" << errno << endl;
                    // ToDo: add appropriate error recovery here. Or ignore it and keep going.
                    // throw("Error reading command file");
                }
                if (isRunning) {
                    return;
                }
            }
        } else if (isRunning) {
            return;
        } else {
            sleep(0.5);
        }

        // Got one or more lines of text. Separate the individual lines and parse them.

        buf[numRead] = '\0';
        //cout << "heard " << buf << endl;

        string lines(buf);
        istringstream sss(lines);

        while (!sss.eof()) {
            char line[512];
            sss.getline(line, sizeof line - 1);
            istringstream ss(line);
            string token;

            ss >> token;
            if (token == "alpha") {
                ss >> alpha;
                cout << "Set alpha=" << alpha << endl;
            }
            else if (token == "eta") {
                ss >> eta;
                cout << "Set eta=" << eta << endl;
            }
            else if (token == "dynamiceta") {
                //ss >> dynamicEtaAdjust;
                ss >> token;
                dynamicEtaAdjust = (token == "True");
                cout << "dynamicEtaAdjust=" << dynamicEtaAdjust << endl;
            }
            else if (token == "lambda") {
                ss >> lambda;
                cout << "Set lambda=" << lambda << endl;
            }
            else if (token == "load") {
                ss >> token;
                cout << "Load weights from " << token << endl;
                loadWeights(token);
            }
            else if (token == "pause") {
                isRunning = false;
                cout << "Pause" << endl;
                sleep(0.5);
            }
            else if (token == "report") {
                //int everyNth;
                //ss >> everyNth;
                ss >> reportEveryNth;
                //setReportingInterval(everyNth);
                //cout << "Report everyNth=" << getReportingInterval() << endl;
                cout << "Report everyNth=" << reportEveryNth << endl;
            }
            else if (token == "averageover") {
                    ss >> recentAverageSmoothingFactor;
                    cout << "Average window over " << recentAverageSmoothingFactor << endl;
            }
            else if (token == "run" || token == "resume") {
                isRunning = true;
                cout << "Resume run" << endl;
            }
            else if (token == "save") {
                ss >> token;
                cout << "Save weights to " << token << endl;
                saveWeights(token);
            }
            else if (token == "repeat") {
                ss >> token;
                repeatInputSamples = (token == "True");
                cout << "repeatInputSamples=" << repeatInputSamples << endl;
            }
            else if (token == "shuffle") {
                ss >> token;
                shuffleInputSamples = (token == "True");
                cout << "shuffleInputSamples=" << shuffleInputSamples << endl;
            }
        }
    } while (!isRunning);
}


// Calculate a new eta parameter based on the current and last average net error.
//
double Net::adjustedEta(void)
{
    const double thresholdUp = 0.01;       // Ignore error increases less than this magnitude
    const double thresholdDown = 0.01;     // Ignore error decreases less than this magnitude
    const double factorUp = 1.01;          // Factor to incrementally increase eta
    const double factorDown = 0.99;        // Factor to incrementally decrease eta

    if (!dynamicEtaAdjust) {
        return eta;
    }

    assert(thresholdUp > 0.0 && thresholdDown > 0.0 && factorUp >= 1.0 && factorDown >= 0.0 && factorDown <= 1.0);

    double errorGradient = (recentAverageError - lastRecentAverageError) / recentAverageError;
    if (errorGradient > thresholdUp) {
        eta = factorDown * eta;
    } else if (errorGradient < -thresholdDown) {
        eta = factorUp * eta;
    }

    return eta;
}


} // end namespace NNet
