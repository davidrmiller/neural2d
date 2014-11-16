/*
neural-net.cpp
David R. Miller, 2014
https://github.com/davidrmiller/neural2d

See neural2d.h for more information.
*/


#include "neural2d.h"
#include <cctype>
#include <utility>  // for pair()

namespace NNet {


//  ***********************************  Utility functions  ***********************************


// Returns a random float in the range [0.0..1.0]
//
float randomFloat(void)
{
    return (float)rand() / RAND_MAX;
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
float elliptDist(float x, float y, float radiusX, float radiusY)
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
float absd(float a) { return a < 0.0 ? -a : a; }


// Extracts the X and Y components from the string of the form "XxY".
// E.g., "8x2" will return the pair (8, 2). "8" returns the pair (8, 1).
//
pair<uint32_t, uint32_t> extractTwoNums(const string &s)
{
    uint32_t sizeX = 0;
    uint32_t sizeY = 0;
    char delim = '\0';

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


// Replaces potentially dangerous chars with underscores
//
void sanitizeFilename(string &s)
{
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (!isalnum(c) && c != '_' && c != '-' && c != '.' && c != '%') {
            s[i] = '_';
        }
    }
}

// Message queue
// A Thread-safe non-blocking FIFO; pushes to the back, pops from the front.
// If the queue is empty, pop() immediately returns with s set to an empty string.

void MessageQueue::push(Message_t &msg)
{
    unique_lock<mutex> locker(mmutex);
    mqueue.push(msg);
}

void MessageQueue::pop(Message_t &msg)
{
    unique_lock<std::mutex> locker(mmutex);
    if (mqueue.empty()) {
        msg.text = "";
        msg.httpResponseFileDes = -1;
    } else {
        msg = mqueue.front();
        mqueue.pop();
    }
}


//  ***********************************  Web server  ***********************************

WebServer::WebServer(void)
{
    socketFd = -1;
    firstAccess = true;        // Used to detect when the first HTTP GET arrives
    initializeHttpResponse();
}

WebServer::~WebServer(void)
{
    stopServer();
}

void WebServer::stopServer(void)
{
    if (socketFd != -1) {
        cout << "Closing webserver socket..." << endl;
        shutdown(socketFd, SHUT_RDWR);
        close(socketFd);
        socketFd = -1;
    }
}


// The web server runs in a separate thread. It receives a GET from the browser,
// extracts the message (e.g., "eta=0.1") and appends it to a shared message
// queue. Periodically, the Net object will pop messages and respond to them
// by constructing a parameter block (a string) and calling a function to
// send the http response.
//
void WebServer::initializeHttpResponse(void)
{
    // Keep appending lines to firstPart until we get the line with the sentinel:
    ifstream httpTemplate("http-response-template.txt");
    if (!httpTemplate.is_open()) {
        firstPart = "Cannot open file \"http-response-template.txt\"; web server is disabled.\r\n";
        return;
    }

    firstPart = "";
    secondPart = "";

    int part = 1; // The lines before the sentinel goes into firstPart

    while (httpTemplate.good()) {
        string line;
        getline(httpTemplate, line);

        // Add \r\n; the \r may already be there:

        if (line[line.length() - 1] == '\r') {
            line.append("\n");
        } else {
            line.append("\r\n");
        }

        // Save the line:

        if (part == 1) {
            firstPart.append(line);
        } else {
            secondPart.append(line);
        }

        // If this is the line with the sentinel, we'll change to secondPart for the rest:

        if (line.find("Parameter block") != string::npos) {
            part = 2;
        }
    }
}


// This sends a 404 Not Found reply
//
void WebServer::replyToUnknownRequest(int httpConnectionFd)
{
    string response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n";

    send(httpConnectionFd, response.c_str(), response.length(), 0);

    shutdown(httpConnectionFd, SHUT_RDWR);
    close(httpConnectionFd);
}

// Look for "POST /" or "GET /?" followed by a message or POST.
// If this is the first GET or POST we receive, we'll treat it as if it were
// "GET / " of the root document.
//
void WebServer::extractAndQueueMessage(string s, int httpConnectionFd, MessageQueue &messages)
{
    struct Message_t msg;
    size_t pos;

    if (firstAccess) {
        msg.text = ""; // Causes a default web form html page to be sent back
        firstAccess = false;
    } else if ((pos = s.find("POST /")) != string::npos) {
        // POST
        pos = s.find("\r\n\r\n");
        assert(pos != string::npos);
        msg.text = s.substr(pos + 4);
    } else if ((pos = s.find("GET /?")) != string::npos) {
        // GET with an argument
        string rawMsg = s.substr(pos + 6);
        pos = rawMsg.find(" ");
        assert(pos != string::npos);
        msg.text = rawMsg.substr(0, pos);
    } else if ((pos = s.find("GET /")) != string::npos) {
        // GET without an argument
        msg.text = "";
    } else {
        // Unknown HTTP request
        replyToUnknownRequest(httpConnectionFd);
        return;
    }

    //cout << "msg.text = " << msg.text << endl;

    msg.httpResponseFileDes = httpConnectionFd;
    messages.push(msg);
}


void WebServer::start(int portNumber_, MessageQueue &messages)
{
    portNumber = portNumber_;
    thread webThread(&WebServer::webServerThread, this, portNumber, ref(messages));
    webThread.detach();
}


// The argument parameterBlock needs to be passed by value because it comes from
// a different thread:
//
void WebServer::sendHttpResponse(string parameterBlock, int httpResponseFileDes)
{
    string response = firstPart + parameterBlock + secondPart;
    const char *buf = response.c_str();

    size_t lenToSend = response.length();
    while (lenToSend > 0) {
        int numWritten = write(httpResponseFileDes, buf, lenToSend);
        if (numWritten < 0) {
            break; // Serious error
        }
        lenToSend -= numWritten;
        buf += numWritten;
    }

    shutdown(httpResponseFileDes, SHUT_RDWR);
    close(httpResponseFileDes);
}


void WebServer::webServerThread(int portNumber, MessageQueue &messages)
{
    static const struct sockaddr_in zero_sockaddr_in = { 0 };
    struct sockaddr_in stSockAddr = zero_sockaddr_in;
    char buff[2048];

    socketFd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (-1 == socketFd) {
        cerr << "Cannot create socket for the web server interface" << endl;
        //exit(1);
        return;
    }

    stSockAddr.sin_family = AF_INET;
    stSockAddr.sin_port = htons(portNumber);
    stSockAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (-1 == bind(socketFd, (struct sockaddr *)&stSockAddr, sizeof(stSockAddr))) {
        cerr << "Cannot bind socket for the web server interface\n\n" << endl;
        close(socketFd);
        exit(1);
        return;
    }

    if (-1 == listen(socketFd, 10)) {
        cerr << "Web server network failure" << endl;
        close(socketFd);
        //exit(1);
        return;
    }

    cout << "Web server started." << endl;

    while (true) {
        int httpConnectionFd = accept(socketFd, NULL, NULL);

        if (0 > httpConnectionFd) {
            cerr << "Webserver failed to accept connection" << endl;
            close(socketFd);
            //exit(1);
            return;
        }

        uint32_t numChars = read(httpConnectionFd, buff, sizeof buff - 1);

        if (numChars > 0) {
            assert(numChars < sizeof buff);
            buff[numChars] = '\0';
            extractAndQueueMessage(buff, httpConnectionFd, messages);
        }
    }

    close(socketFd);
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
        cout << "Undefined transfer function: \'" << transferFunctionName << "\'" << endl;
        throw("Error in topology config file: undefined transfer function");
    }
}


layerParams_t::layerParams_t(void)
{
    clear();
}

void layerParams_t::clear(void)
{
    layerName.clear();
    fromLayerName.clear();
    sizeX = 0;
    sizeY = 0;
    channel = NNet::BW;
    colorChannelSpecified = false;
    radiusX = 1e9;
    radiusY = 1e9;
    transferFunctionName.clear();
    tf = transferFunctionTanh;
    tfDerivative = transferFunctionDerivativeTanh;
    convolveMatrix.clear();       // Format: convolve {{0,1,0},...}
    isConvolutionLayer = false;   // Equivalent to (convolveMatrix.size() != 0)
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
        cout << "Error reading image file \'" << filename << "\'" << endl;
        // To do: add appropriate error recovery here
        throw "Error reading input image file";
    }

    // Read the BMP header to get the image dimensions:

    unsigned char info[54];
    if (fread(info, sizeof(unsigned char), 54, f) != 54) {
        cout << "Error reading the image header from \'" << filename << "\'" << endl;
        throw "Error reading input image file";
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
                throw("Error: unknown pixel conversion");
            }

            // Convert it to the range 0.0..1.0: this value will be the input to an input neuron:
            dataContainer.push_back(val / 256.0);
        }
    }

    fclose(f);
    delete[] imageData;
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
// in memory. For now, we'll just read the image filenames and, if available, the
// target output values. We'll defer reading the image pixel data until it's needed.

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
        string token;
        char delim;

        stringstream ss(line);
        ss >> token;
        if (token == "{") {
            // This means we have literal values like "{ 0.2 0 -1.0}"
            sample.imageFilename="";   // "" means we have immediate data

            // Read from the char after the { up to but including the } char:
            char args[1024]; // Review !!!
            ss.get(args, sizeof args, '}');
            stringstream inargs(args);
            while (!inargs.eof()) {
                float val;
                if (!(inargs >> val).fail()) {
                    sample.data.push_back(val);
                }
            }
            ss >> delim;
        } else {
            ss >> sample.imageFilename;
            sample.imageFilename = token + sample.imageFilename; // uugh - refactor this block
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

    cout << samples.size() << " training samples initialized" << endl;
}


// Randomize the order of the samples container.
//
void SampleSet::shuffle(void)
{
    random_shuffle(samples.begin(), samples.end());
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
Neuron::Neuron(vector<Connection> *pConnectionsData, transferFunction_t tf, transferFunction_t tfDerivative)
{
    //output = randomFloat() - 0.5;
    gradient = 0.0;
    pConnections = pConnectionsData; // Remember where the connections array is
    backConnectionsIndices.clear();
    forwardConnectionsIndices.clear();
    sourceNeurons.clear();

    transferFunction = tf;
    transferFunctionDerivative = tfDerivative;
}


// The error gradient of an output-layer neuron is equal to the target (desired)
// value minus the computed output value, times the derivative of
// the output-layer activation function evaluated at the computed output value.
//
void Neuron::calcOutputGradients(float targetVal)
{
    float delta = targetVal - output;
    gradient = delta * Neuron::transferFunctionDerivative(output);
}


// The error gradient of a hidden-layer neuron is equal to the derivative
// of the activation function of the hidden layer evaluated at the
// local output of the neuron times the sum of the product of
// the primary outputs times their associated hidden-to-output weights.
//
void Neuron::calcHiddenGradients(void)
{
    float dow = sumDOW_nextLayer();
    gradient = dow * Neuron::transferFunctionDerivative(output);
}


// To do: add commentary!!!
//
float Neuron::sumDOW_nextLayer(void) const
{
    float sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (auto idx : forwardConnectionsIndices) {
        const Connection &conn = (*pConnections)[idx];

        sum += conn.weight * conn.toNeuron.gradient;
    }

    return sum;
}


void Neuron::updateInputWeights(float eta, float alpha)
{
    // The weights to be updated are the weights from the neurons in the
    // preceding layer (the source layer) to this neuron:

    for (auto idx : backConnectionsIndices) {
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


// To feed forward an individual neuron, we'll sum the weighted inputs, then pass that
// sum through the transfer function.
//
void Neuron::feedForward(void)
{
    float sum = 0.0;

    // Sum the neuron's inputs:
    for (size_t i = 0; i < backConnectionsIndices.size(); ++i) {
        size_t idx = backConnectionsIndices[i];
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
    sumWeights = 0.0;
    enableWebServer = true;
    portNumber = 24080;

    // Web interface:

    webServer.start(portNumber, messages);

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


Net::~Net(void) {
    webServer.stopServer();
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
        for (float targetVal : sample.targetVals) {
            cout << targetVal << " ";
        }

        // Optional: Enable the following block if you would like to report the net's
        // outputs as a classifier, where the output neuron with the largest output
        // value indicates which class was recognized. This can be used, e.g., for pattern
        // recognition where each output neuron corresponds to one pattern class,
        // and the output neurons are trained to be high to indicate a pattern match,
        // and low to indicate no match.

        if (false) {
            //float maxOutput = (numeric_limits<float>::min)();
            float maxOutput = -1.e8;
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
        cout << "  eta=" << eta << " ";

        // Show overall net error for this sample and for the last few samples averaged:
        cout << "Net error = " << error << ", running average = " << recentAverageError << endl;
    }
}


// Given an existing layer with neurons already connected, add more
// connections. This happens when a layer specification is repeated in
// the config file, thus creating connections to source neurons from
// multiple layers. This applies to regular neurons and neurons in a
// convolution layer. Returns false for any error, true if successful.
//
bool Net::addToLayer(Layer &layerTo, Layer &layerFrom,
        layerParams_t &params)
{
    if (params.sizeX != layerTo.params.sizeX || params.sizeY != layerTo.params.sizeY) {
        cerr << "Error: Config: repeated layer '" << layerTo.params.layerName
             << "' must be the same size" << endl;
        return false;
    }

    // Layer already exists -- add more backConnections to the existing layer:

    for (uint32_t ny = 0; ny < layerTo.params.sizeY; ++ny) {
        cout << "\r" << ny << flush; // Progress indicator

        for (uint32_t nx = 0; nx < layerTo.params.sizeX; ++nx) {
            //cout << "connect to neuron " << nx << "," << ny << endl;
            connectNeuron(layerTo, layerFrom, layerTo.neurons[flattenXY(nx, ny, layerTo.params.sizeX)],
                          nx, ny, params);
            // n.b. Bias connections were already made when the neurons were first created.
        }
    }

    cout << endl; // End progress indicator

    return true;
}


// Given a layer name and size, create an empty layer. No neurons are created yet.
//
Layer &Net::createLayer(const layerParams_t &params)
{
    layers.push_back(Layer());
    Layer &layer = layers.back(); // Make a convenient name

    layer.params = params;

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

    cout << "\n\nNet configuration (incl. bias connection): --------------------------" << endl;

    for (auto const &l : layers) {
        numFwdConnections = 0;
        numBackConnections = 0;
        cout << "Layer '" << l.params.layerName << "' has " << l.neurons.size()
             << " neurons arranged in " << l.params.sizeX << "x" << l.params.sizeY << ":" << endl;

        for (auto const &n : l.neurons) {
            if (details) {
                cout << "  neuron(" << &n << ")" << " output: " << n.output << endl;
            }

            numFwdConnections += n.forwardConnectionsIndices.size();
            numBackConnections += n.backConnectionsIndices.size(); // Includes the bias connection

            if (details && n.forwardConnectionsIndices.size() > 0) {
                cout << "    Fwd connections:" << endl;
                for (auto idx : n.forwardConnectionsIndices) {
                    Connection const &pc = connections[idx];
                    cout << "      conn(" << &pc << ") pFrom=" << &pc.fromNeuron
                         << ", pTo=" << &pc.toNeuron 
                         << ", w,dw=" << pc.weight << ", " << pc.deltaWeight
                         << endl;
                }
            }

            if (details && n.backConnectionsIndices.size() > 0) {
                cout << "    Back connections (incl. bias):" << endl;
                for (auto idx : n.backConnectionsIndices) {
                    Connection const &c = connections[idx];
                    cout << "      conn(" << &c << ") pFrom=" << &c.fromNeuron
                         << ", pTo=" << &c.toNeuron
                         << ", w=" << c.weight

                         << endl;
                    assert(&c.toNeuron == &n);
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
    // update connection weights for regular neurons. Skip the udpate in
    // convolution layers.

    for (uint32_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = layers[layerNum];

    // Optionally enable the following #pragma line to permit OpenMP to
    // parallelize this loop. For gcc 4.x, add the option "-fopenmp" to
    // the compiler command line. If the compiler does not understand
    // OpenMP, the #pragma will be ignored.

#pragma omp parallel for

        for (size_t i = 0; i < layer.neurons.size(); ++i) {
            Neuron &neuron = layer.neurons[i];
            if (!layer.params.isConvolutionLayer) {
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
        cout << "Error: input sample " << inputSampleNumber << " has " << data.size()
             << " components, expecting " << inputLayer.neurons.size() << endl;
        //throw("Wrong number of inputs");
    }

    // Rather than make it a fatal error if the number of input neurons != number
    // of input data values, we'll use whatever we can and skip the rest:
    // To do: report a mismatch in number of inputs neurons and size of input sample!!!

    for (uint32_t i = 0; i < (uint32_t)min(inputLayer.neurons.size(), data.size()); ++i) {
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
        float delta = sample.targetVals[n] - outputLayer.neurons[n].output;
        error += delta * delta;
    }

    error /= 2.0 * outputLayer.neurons.size();

    // Regularization calculations -- if this experiment works, calculate the sum of weights on the fly
    // during backprop for performance reasons:

    float sumWeightsSquared = 0.0;
    if (lambda != 0.0) {
        // For all layers except the input layer, sum all the weights. These are in
        // the back-connection records:
        for (uint32_t layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
            Layer &layer = layers[layerIdx];
            for (auto const &neuron : layer.neurons) {
                // To do: skip the bias connection correctly;
                for (uint32_t idx = 0; idx < neuron.backConnectionsIndices.size() - 1; ++idx) {
                    Connection &conn = (*neuron.pConnections)[neuron.backConnectionsIndices[idx]];
                    sumWeightsSquared += conn.weight * conn.weight;
                }
            }
        }

        float sumWeights = (sumWeightsSquared * lambda)
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
//
// Neurons can be "regular" neurons, or convolution nodes. If a convolution matrix is
// defined for the layer, the neurons in that layer will be connected to source neurons
// in a rectangular pattern defined by the matrix dimensions. No bias connections are
// created for convolution nodes. Convolution nodes ignore any radius parameter.
// For convolution nodes, the transfer function is set to be the identity function.
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
        uint32_t nx, uint32_t ny, layerParams_t &params)
{
    uint32_t sizeX = layerTo.params.sizeX;
    uint32_t sizeY = layerTo.params.sizeY;
    assert(sizeX > 0 && sizeY > 0);

    // Calculate the normalized [0..1] coordinates of our neuron:
    float normalizedX = ((float)nx / sizeX) + (1.0 / (2 * sizeX));
    float normalizedY = ((float)ny / sizeY) + (1.0 / (2 * sizeY));

    // Calculate the coords of the nearest neuron in the "from" layer.
    // The calculated coords are relative to the "from" layer:
    uint32_t lfromX = uint32_t(normalizedX * fromLayer.params.sizeX); // should we round off instead of round down?
    uint32_t lfromY = uint32_t(normalizedY * fromLayer.params.sizeY);

//    cout << "our neuron at " << nx << "," << ny << " covers neuron at "
//         << lfromX << "," << lfromY << endl;

    // Calculate the rectangular window into the "from" layer:

    int32_t xmin = 0;
    int32_t xmax = 0;
    int32_t ymin = 0;
    int32_t ymax = 0;

    if (params.isConvolutionLayer) {
        assert(params.convolveMatrix.size() > 0);
        assert(params.convolveMatrix[0].size() > 0);

        ymin = lfromY - params.convolveMatrix[0].size() / 2;
        ymax = ymin + params.convolveMatrix[0].size() - 1;
        xmin = lfromX - params.convolveMatrix.size() / 2;
        xmax = xmin + params.convolveMatrix.size() - 1;
    } else {
        xmin = lfromX - params.radiusX;
        xmax = lfromX + params.radiusX;
        ymin = lfromY - params.radiusY;
        ymax = lfromY + params.radiusY;
    }

    // Clip to the layer boundaries:

    if (xmin < 0) xmin = 0;
    if (xmin >= (int32_t)fromLayer.params.sizeX) xmin = fromLayer.params.sizeX - 1;
    if (ymin < 0) ymin = 0;
    if (ymin >= (int32_t)fromLayer.params.sizeY) ymin = fromLayer.params.sizeY - 1;
    if (xmax < 0) xmax = 0;
    if (xmax >= (int32_t)fromLayer.params.sizeX) xmax = fromLayer.params.sizeX - 1;
    if (ymax < 0) ymax = 0;
    if (ymax >= (int32_t)fromLayer.params.sizeY) ymax = fromLayer.params.sizeY - 1;

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
            if (!params.isConvolutionLayer && !projectRectangular && elliptDist(xcenter - x, ycenter - y,
                                                  params.radiusX, params.radiusY) >= 1.0) {
                continue; // Skip this location, it's outside the ellipse
            }

            if (params.isConvolutionLayer && params.convolveMatrix[x - xmin][y - ymin] == 0.0) {
                // Skip this connection because the convolve matrix weight is zero:
                continue;
            }

            Neuron &fromNeuron = fromLayer.neurons[flattenXY(x, y, fromLayer.params.sizeX)];

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
                ++totalNumberConnections;

                // Initialize the weight of the connection:
                if (params.isConvolutionLayer) {
                    connections.back().weight = params.convolveMatrix[x - xmin][y - ymin];
                } else {
                    //connections.back().weight = (randomFloat() - 0.5) / maxNumSourceNeurons;
                    connections.back().weight = ((randomFloat() * 2) - 1.0) / sqrt(maxNumSourceNeurons);
                }

                // Record the back connection index at the destination neuron:
                neuron.backConnectionsIndices.push_back(connectionIdx);

                // Remember the source neuron for detecting duplicate connections:
                neuron.sourceNeurons.insert(&fromNeuron);

                // Record the Connection index at the source neuron:
                uint32_t flatIdxFrom = flattenXY(x, y, fromLayer.params.sizeX); // x * fromLayer.sizeY + y;
                fromLayer.neurons[flatIdxFrom].forwardConnectionsIndices.push_back(
                           connectionIdx);

//                cout << "    connect from layer " << fromLayer.params.layerName
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
    c.weight = randomFloat() - 0.5; // Review this !!!
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
    for (uint32_t ln = 0; ln < layers.size(); ++ln) {
        if (layers[ln].params.layerName == name) {
            return (int32_t)ln;
        }
    }

    return -1;
}


// Create neurons and connect them. For the input layer, there are no incoming
// connections and radius doesn't apply. Calling this function with layerFrom == layerTo
// indicates an input layer.
//
void Net::createNeurons(Layer &layerTo, Layer &layerFrom, layerParams_t &params)
{
    // Reserve enough space in layer.neurons to prevent reallocation (so that
    // we can form stable references to neurons):

    layerTo.neurons.reserve(layerTo.params.sizeX * layerTo.params.sizeY);

    for (uint32_t ny = 0; ny < layerTo.params.sizeY; ++ny) {

        cout << "\r" << ny << "/" << layerTo.params.sizeY << flush; // Progress indicator

        for (uint32_t nx = 0; nx < layerTo.params.sizeX; ++nx) {
            // When we create a neuron, we have to give it a pointer to the
            // start of the array of Connection objects:
            layerTo.neurons.push_back(Neuron(&connections, params.tf, params.tfDerivative));
            Neuron &neuron = layerTo.neurons.back(); // Make a more convenient name
            ++totalNumberNeurons;

            // If layerFrom is layerTo, it means we're making input neurons
            // that have no input connections to the neurons. Else, we must make connections
            // to the source neurons and, for classic neurons, to a bias input:

            if (&layerFrom != &layerTo) {
                connectNeuron(layerTo, layerFrom, neuron,
                              nx, ny, params);
                if (!params.isConvolutionLayer) {
                    connectBias(neuron);
                }
            }
        }
    }

    cout << endl; // End the progress indicator
}


/*
Convoluation matrixExamples:
{0, 1,2}
{ {0,1,2}, {1,2,1}, {0, 1, 0}}
*/
convolveMatrix_t Net::parseMatrixSpec(istringstream &ss)
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
            throw("Internal error in parsing convolve matrix spec");
        }

        action_t action = table[lastState][newState];

        switch(action) {
        case SKIP:
            break;
        case ILL:
            cout << "Error in convolve matrix spec" << endl;
            throw("Syntax error in convolve matrix spec");
            break;
        case PLINC:
            ++braceLevel;
            break;
        case PLDECX:
            --braceLevel;
            if (braceLevel != 0) {
                cout << "Error in convolve matrix spec" << endl;
                throw("Syntax error in convolve matrix spec");
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
        }
        convMat.push_back(vector<float>());
        for (unsigned y = 0; y < mat[x].size(); ++y) {
            convMat.back().push_back(mat[x][y]);
        }
        if (convMat.back().size() != firstRowSize) {
            cout << "Warning: in convolution matrix in topology config file, inconsistent matrix row size" << endl;
            throw("Error in topology config file: inconsistent row size in convolve matrix spec");
        }
    }

    return convMat;
}


// Returns true if the neural net was successfully created and connected. Returns
// false for any error. See the GitHub wiki (https://github.com/davidrmiller/neural2d)
// for more information about the format of the topology config file.
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

    // params will hold the parameters extracted from each config line:
    layerParams_t params;

    // Reset the stream pointer and parse the config file for real now:

    int32_t previouslyDefinedLayerNumSameName;
    uint32_t numNeurons = 0;
    lineNum = 0;
    cfg.clear();   // Reset the stream EOF
    cfg.seekg(0);  // Rewind the stream, start over

    while (getline(cfg, line)) {
        ++lineNum;
        istringstream ss(line);
        params.clear();

        bool done = false;
        while (!done && !ss.eof()) {
            ss >> params.layerName;  // First field is the layer name

            // Skip blank and comment lines:
            if (params.layerName == "" || params.layerName[0] == '#') {
                done = true;
                continue;
            }

            // There will be one or more additional fields:

            string tempString;
            string token;
            char delim;

            while (!ss.eof()) {
                pair<uint32_t, uint32_t> twoNums;
                token.clear();
                tempString.clear();
                delim = '\0';

                ss >> token;
                if (token == "") {
                    continue;
                } else if (token == "from") {
                    ss >> params.fromLayerName;
                } else if (token == "size") {
                    ss >> tempString;
                    twoNums = extractTwoNums(tempString);
                    params.sizeX = twoNums.first;
                    params.sizeY = twoNums.second;
                } else if (token == "channel") {
                    ss >> tempString;  // Expecting R, G, B, or BW
                    if      (tempString == "R")  params.channel = NNet::R;
                    else if (tempString == "G")  params.channel = NNet::G;
                    else if (tempString == "B")  params.channel = NNet::B;
                    else if (tempString == "BW") params.channel = NNet::BW;
                    else {
                        throw("Unknown color channel");
                    }
                    params.colorChannelSpecified = true;
                } else if (token == "radius") {
                    ss >> params.radiusX >> delim >> params.radiusY;
                    if (delim == '\0') {
                        params.radiusY = 0;
                    }
                } else if (token == "tf") {
                    ss >> params.transferFunctionName;
                    params.resolveTransferFunctionName();
                } else if (token == "convolve") {
                    params.convolveMatrix = parseMatrixSpec(ss);
                    if (ss.tellg() == -1) {
                        break; // todo: figure out why this break is necessary
                    }
                    params.isConvolutionLayer = true;
                    params.transferFunctionName = "identity";
                    params.resolveTransferFunctionName();
                } else {
                    cout << "Syntax error in " << configFilename << " line " << lineNum << endl;
                    throw("Topology config file syntax error");
                }
            }

            // Now check the data we extracted:

            // If this is the first layer, check that it is called "input" and that there
            // is no convolution matrix defined:

            if (layers.size() == 0 && params.layerName != "input") {
                cerr << "Error in " << configFilename << ": the first layer must be named 'input'" << endl;
                throw("Topology config file syntax error");

                if (params.convolveMatrix.size() > 0) {
                    cerr << "Error in " << configFilename << ": input layer cannot be a convolution layer" << endl;
                    throw("Topology config file syntax error");
                }
            }

            // If this is not the first layer, check that it does not have a channel parameter
            // and that there is no conflicting tf or radius specified with a convolve matrix.

            if (layers.size() != 0 && params.layerName != "input") {
                if (params.colorChannelSpecified) {
                    throw("Error in topology config file: color channel applies only to input layer");
                }
            }

            // Check the source layer:

            int32_t layerNumFrom = getLayerNumberFromName(params.fromLayerName); // input layer will return -1

            if (params.layerName != "input") {
                // If not input layer, then the source layer must already exist:
                if (layerNumFrom == -1) {
                    cerr << "Error: Config(" << lineNum << "): from layer '"
                         << params.fromLayerName << "' is undefined." << endl;
                    throw("Topology config file syntax error");
                }
            } else {
                // For input layer, from layer is not allowed:
                if (params.fromLayerName != "") {
                    cerr << "Error: Config(" << lineNum << "): "
                         << " input layer cannot have a 'from' parameter" << endl;
                    throw("Topology config file syntax error");
                }
            }

            // Check if a layer already exists with the same name:

            previouslyDefinedLayerNumSameName = getLayerNumberFromName(params.layerName);
            if (previouslyDefinedLayerNumSameName == -1) {

                // To do: Add range check for sizeX, sizeY, radiusX, radiusY, convolveMatrix params !!!

                // Create a new layer of this name.
                // "input" layer will always take this path.

                Layer &newLayer = createLayer(params);

                // Create neurons and connect them:

                cout << "Creating layer " << params.layerName << ", one moment..." << endl;
                if (params.layerName == "input") {
                    createNeurons(newLayer, newLayer, params); // Input layer has no back connections
                } else {
                    createNeurons(newLayer, layers[layerNumFrom], params); // Also connects them
                }

                numNeurons += params.sizeX * params.sizeY;
            } else {

                // Layer already exists, add connections to it.
                // "input" layer will never take this path.

                Layer &layerToRef = layers[previouslyDefinedLayerNumSameName]; // A more convenient name
                if (layerToRef.params.sizeX != params.sizeX || layerToRef.params.sizeY != params.sizeY) {
                    cerr << "Error: Config(" << lineNum << "): layer '" << params.layerName
                         << "' already exists, but different size" << endl;
                    throw("Topology config file syntax error");
                }

                // Add more connections to the existing neurons in this layer:

                bool ok = addToLayer(layerToRef, layers[layerNumFrom], params);
                if (!ok) {
                    cout << "Error in " << configFilename << ", layer \'" << params.layerName << "\'" << endl;
                    throw("Error adding connections");
                }
            }
        }
    }

    // Check that the last layer defined is named "output":

    if (layers.back().params.layerName != "output") {
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
                cout << "  neuron(" << &neuron << ") on " << layer.params.layerName
                     << endl;
            }
        }
    }

    // Optionally enable the next line to display the resulting net topology:
    debugShowNet(false);

    cout << "\nConfig file parsed successfully." << endl;
    cout << "Found " << neuronsWithNoSink << " neurons with no sink." << endl;
    cout << numNeurons << " neurons total; " << totalNumberConnections << "=" << connections.size()
         << " back+bias connections." << endl;
    cout << "About " << (int)((float)totalNumberConnections / numNeurons + 0.5)
         << " connections per neuron on average." << endl;
    cout << (isRunning ? "" : "Paused.") << endl;
}


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


// Handler for the optional external controller for the neural2d program.
// This function reads the command file and acts on any commands received.
//
void Net::actOnMessageReceived(Message_t &msg)
{
    string parameterBlock;
    ColorChannel_t newColorChannel = layers[0].params.channel;

    string &line = msg.text;

    //cout << "acting on message: \"" << line << "\"" << endl;

    if (line == "" &&  msg.httpResponseFileDes != -1) {
        makeParameterBlock(parameterBlock);
        webServer.sendHttpResponse(parameterBlock, msg.httpResponseFileDes);
        return;
    }

    istringstream ss(line);
    string token;

    ss >> token;

    // trainShadow

    if (token.find("trainShadow=&train=on") == 0) {
        enableBackPropTraining = true;
    } else if (token.find("trainShadow=") == 0) {
        enableBackPropTraining = false;
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
    } else if (token.find("channelGShadow=") == 0) {
        newColorChannel = NNet::G;
    } else if (token.find("channelBShadow=") == 0) {
        newColorChannel = NNet::B;
    } else if (token.find("channelBWShadow=") == 0) {
        newColorChannel = NNet::BW;
    }

    // alpha

    else if (token.find("alpha=") == 0) {
        alpha = strtod(token.substr(6).c_str(), NULL);
        cout << "Set alpha=" << alpha << endl;
    }

    else if (token.find("eta=") == 0) {
        eta = strtod(token.substr(4).c_str(), NULL);
        cout << "Set eta=" << eta << endl;
    }

    else if (token.find("etaShadow=&dynamicEta=1") == 0) {
        dynamicEtaAdjust = true;
        cout << "dynamicEtaAdjust=" << dynamicEtaAdjust << endl;
    }

    else if (token.find("etaShadow=") == 0) {
        dynamicEtaAdjust = false;
        cout << "dynamicEtaAdjust=" << dynamicEtaAdjust << endl;
    }

    else if (token.find("lambda") == 0) {
        lambda = strtod(token.substr(6).c_str(), NULL);
        cout << "Set lambda=" << lambda << endl;
    }

    else if (token == "load") {
        ss >> token;
        cout << "Load weights from " << token << endl;
        loadWeights(token);
    }

    else if (token.find("pause") == 0) {
        isRunning = false;
        cout << "Pause" << endl;
        sleep(0.5);
    }

    else if (token.find("reportEveryNth=") == 0) {
        reportEveryNth = strtod(token.substr(15).c_str(), NULL);
        cout << "Report everyNth=" << reportEveryNth << endl;
    }

    else if (token.find("smoothingFactor=") == 0) {
        recentAverageSmoothingFactor = strtod(token.substr(16).c_str(), NULL);
        cout << "Average window over " << recentAverageSmoothingFactor << endl;
    }

    else if (token.find("weightsFile=") == 0) {
        weightsFilename = token.substr(12);
        sanitizeFilename(weightsFilename);
        cout << "weightsFilename = " << weightsFilename << endl;
    }

    else if (token == "run" || token.find("resume") == 0) {
        isRunning = true;
        cout << "Resume run" << endl;
    }

    else if (token.find("savew") == 0) {
        cout << "Save weights to " << weightsFilename << endl;
        saveWeights(weightsFilename);
    }

    else if (token.find("loadw") == 0) {
        cout << "Load weights from " << weightsFilename << endl;
        loadWeights(weightsFilename);
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


void Net::doCommand()
{
    // First check the web interface:

    do {
        Message_t msg;
        messages.pop(msg);
        if (msg.httpResponseFileDes != -1) {
            actOnMessageReceived(msg);
        }
        if (!isRunning) {
            sleep(0.5);
        }
    } while (!isRunning);
}


// Calculate a new eta parameter based on the current and last average net error.
//
float Net::adjustedEta(void)
{
    const float thresholdUp = 0.01;       // Ignore error increases less than this magnitude
    const float thresholdDown = 0.01;     // Ignore error decreases less than this magnitude
    const float factorUp = 1.01;          // Factor to incrementally increase eta
    const float factorDown = 0.99;        // Factor to incrementally decrease eta

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
