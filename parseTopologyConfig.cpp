/*
This is parseTopologyConfig.cpp, part of neural2d.
This is the part that parses the topology config file.
David R. Miller, 2015
https://github.com/davidrmiller/neural2d

Also see neural2d.h for more information.
*/

#include <fstream>
#include <iostream>
#include <vector>
#include "neural2d.h"

namespace NNet {


topologyConfigSpec_t::topologyConfigSpec_t(void)
{
    fromLayerName = "";
    fromLayerIndex = 0;
    sizeSpecified = false;
    colorChannelSpecified = false;
    radiusSpecified = false;
    tfSpecified = false;

    size.depth = size.x = size.y = 0;
    channel = NNet::BW;
    radius.x = radius.y = 0.0;
    transferFunctionName.clear();
    transferFunctionName = "tanh";
    flatConvolveMatrix.clear();
    isConvolutionFilterLayer = false;   // Equivalent to (convolveMatrix.size() == 1)
    isConvolutionNetworkLayer = false;  // Equivalent to (convolveMatrix.size() > 1)
    isPoolingLayer = false;             // Equivalent to (poolSize.x != 0)
    kernelSize.x = kernelSize.y = 0;    // Used only for convolution filter and conv. network layers
    poolSize.x = poolSize.y = 0;        // Used only for convolution network layers
    poolMethod = POOL_NONE;
}


void configErrorThrow(topologyConfigSpec_t &params, const string &msg)
{
    err << "There's a problem in the topology config file at line " << params.configLineNum << ":";
    if (params.layerName.size() > 0) {
        err << "(layer \"" << params.layerName << "\")";
    }
    err << endl;
    err << msg << endl;
    throw exceptionConfigFile();
}


/*
Convolution filter matrix example formats:
{0, 1,2}
{ {0,1,2}, {1,2,1}, {0, 1, 0}}
*/
void extractConvolveFilterMatrix(topologyConfigSpec_t &params, std::istringstream &ss)
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
            configErrorThrow(params, "Warning: Internal error in parsing convolve filter matrix spec");
        }

        action_t action = table[lastState][newState];

        switch(action) {
        case SKIP:
            break;
        case ILL:
            configErrorThrow(params, "Syntax error in convolve filter matrix spec");
            break;
        case PLINC:
            ++braceLevel;
            break;
        case PLDECX:
            --braceLevel;
            if (braceLevel != 0) {
                configErrorThrow(params, "Syntax error in convolve filter matrix spec");
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

    // Check that all rows have the same size:

    unsigned firstRowSize = mat[0].size();
    if (0 != count_if(mat.begin() + 1, mat.end(), [firstRowSize](vector<float> row) {
             return row.size() != firstRowSize; })) {
        configErrorThrow(params, "Error in topology config file: inconsistent row size in convolve filter matrix spec");
    }

    // We'll create (or recreate) a one-element flatConvolveMatrix in the params structure:
    // Convolution filtering only needs a single convolve matrix, so only element zero is
    // used in params.flatConvolveMatrix. That one element will be a flattened
    // container of the 2D convolve matrix:

    params.flatConvolveMatrix.clear();
    params.flatConvolveMatrix.push_back(vector<float>()); // Start with one empty container

    for (auto &row : mat) {
        for (auto val : row) {
            params.flatConvolveMatrix[0].push_back(val);
        }
    }

    // The matrix is arranged so that we can access elements as [x][y]:
    params.kernelSize.x = mat.size();
    params.kernelSize.y = mat[0].size();
}


// format: [depth *] X [x Y]
// Depth, if omitted, defaults to zero.
// Y, if omitted, defaults to zero.
//
dxySize extractDxySize(std::istringstream &ss)
{
    char ch;
    dxySize size;
    size.depth = 1;  // Default is 1 unless otherwise specified

    ss >> size.x;   // This may actually be the depth, we'll see
    auto pos = ss.tellg();
    ss >> ch;       // Test the next non-space char
    if (ch == '*') {
        // Depth dimension
        size.depth = size.x; // That was the depth we read, not X
        size.x = 0;
        ss >> size.x;
        pos = ss.tellg();
        ss >> ch;       // Test the next non-space char
    }

    if (ch == 'x') {
        ss >> size.y;
    } else {
        ss.seekg(pos);  // Put back what is not ours
        size.y = 1;     // E.g., "8" means 8x1
    }

    return size;
}

// format: X [x Y]
// Y, if omitted, defaults to zero.
//
xySize extractXySize(std::istringstream &ss)
{
    char ch;
    xySize size;

    ss >> size.x;
    auto pos = ss.tellg();
    ss >> ch;  // Test the next non-space char
    if (ch == 'x') {
        ss >> size.y;
    } else {
        ss.seekg(pos);   // Put back what is not ours
        size.y = 1;   // E.g., "8" means 8x1
    }

    return size;
}


// Throws for any error.
//
void extractChannel(topologyConfigSpec_t &params, std::istringstream &ss)
{
    string stoken;
    ss >> stoken;
    if      (stoken == "R")  params.channel = NNet::R;
    else if (stoken == "G")  params.channel = NNet::G;
    else if (stoken == "B")  params.channel = NNet::B;
    else if (stoken == "BW") params.channel = NNet::BW;
    else {
        configErrorThrow(params, "Unknown color channel");
    }

    params.colorChannelSpecified = true;
}


// Throws for any error.
// Modifies params and leaves ss pointing to the next char after the pool method:
//
void extractPoolMethod(topologyConfigSpec_t &params, std::istringstream &ss)
{
    string stoken;

    ss >> stoken;
    if (stoken == "max") {
        params.poolMethod = POOL_MAX;
    } else if (stoken == "avg") {
        params.poolMethod = POOL_AVG;
    } else {
        configErrorThrow(params, "Expected pool method \"max\" or \"avg\"");
    }
}


// Topology config grammar:
//
// layer-name parameters
// parameters := parameter [ parameters ]
// parameter :=
//    input | output | layer-name
//    size dxy-spec
//    from layer-name
//    channel channel-spec
//    radius xy-spec
//    tf transfer-function-spec
//    convolve filter-spec
//    convolve xy-spec
//    pool { max | avg } xy-spec
// dxy-spec := integer * xy-spec
// xy-spec := integer [ x integer ]
// channel-spec := R|G|B|BW
// filter-spec := max|avg
//
// Returns true if we successfully extracted params, else returns
// false for any error or if the line is a comment or blank line:
//
bool extractOneLayerParams(topologyConfigSpec_t &params, const string &line)
{
    string stoken;
    char ctoken;
    std::istringstream ss(line);

    ss >> stoken;  // First token is always the layer name
    if (stoken == "" || stoken[0] == '#') {
        return false;
    }

    params.layerName = stoken;

    // Extract the rest of the parameters:

    bool done = false;
    while (!done && !ss.eof() && ss.tellg() != -1) {
        string stoken;
        ss >> stoken;
        if (stoken.size() == 0) {
            break;
        }

        if (stoken == "size") {
            params.size = extractDxySize(ss);
            params.sizeSpecified = true;
        } else if (stoken == "from") {
            ss >> params.fromLayerName;
        } else if (stoken == "channel") {
            extractChannel(params, ss);
            params.colorChannelSpecified = true;
        } else if (stoken == "radius") {
            params.radius = extractXySize(ss);
            params.radiusSpecified = true;
        } else if (stoken == "tf") {
            ss >> params.transferFunctionName;
            params.tfSpecified = true;
        } else if (stoken == "convolve") {
            // The next non-space char determines whether this is a
            // conv-filter or conv-network parameter:
            auto pos = ss.tellg();
            ss >> ctoken;
            if (ctoken == '{') {
                // Convolution filter spec    expects: {{},{}}
                ss.seekg(pos);
                extractConvolveFilterMatrix(params, ss); // Allocates and initializes the matrix
                params.isConvolutionFilterLayer = true;
            } else {
                extern float randomFloat(void);
                // Convolution network layer  expects: kernel size to be defined
                ss.seekg(pos);
                params.kernelSize = extractXySize(ss);
                // Construct a matrix of the correct size, and make depth copies
                vector<float> flatMatrix(params.kernelSize.x * params.kernelSize.y);
                std::for_each(flatMatrix.begin(), flatMatrix.end(), [](float &w) {
                        w = randomFloat() / 100.0; // !!!
                });
                params.isConvolutionNetworkLayer = true;
                params.flatConvolveMatrix.assign(params.size.depth, flatMatrix);
            }
        } else if (stoken == "pool") {
            extractPoolMethod(params, ss);
            params.poolSize = extractXySize(ss);
            params.isPoolingLayer = true;
        } else {
            configErrorThrow(params, "Unknown parameter");
        }
    }

    return true;
}


// Fix-ups and consistency checks go here after all the records have
// been collected:
//
void consistency(vector<topologyConfigSpec_t> &params)
{
    if (params.size() < 2) {
        err << "Topology config spec needs at least an input and output layer" << endl;
        throw exceptionConfigFile();
    }

    // Specific only to input layer:

    if (params[0].layerName != "input") {
        err << "First layer must be named input" << endl;
        throw exceptionConfigFile();
    }

    if (params[0].fromLayerName.size() > 0) {
        warn << "Input layer cannot have a from parameter" << endl;
    }

    if (params[0].kernelSize.x != 0 || params[0].kernelSize.y != 0) {
        err << "Input layer cannot have a convolve parameter" << endl;
        throw exceptionConfigFile();
    }

    if (params[0].isConvolutionFilterLayer || params[0].isConvolutionNetworkLayer) {
        err << "Input layer cannot have a convolve parameter" << endl;
        throw exceptionConfigFile();
    }

    if (params[0].isPoolingLayer) {
        err << "Input layer cannot have a pool parameter" << endl;
        throw exceptionConfigFile();
    }

    if (params[0].radiusSpecified) {
        err << "Input layer cannot have a radius parameter" << endl;
        throw exceptionConfigFile();
    }

    if (params[0].tfSpecified) {
        err << "Input layer cannot have a tf parameter" << endl;
        throw exceptionConfigFile();
    }

    // In common to hidden layer and output layer specs:

    for (auto it = params.begin() + 1; it != params.end(); ++it) {
        auto &spec = *it;

        // Ensure that if a layer name is repeated with a size, the size must
        // match the size of the previous spec:
        if (spec.sizeSpecified) {
            for (auto itp = it - 1; itp != params.begin() - 1; --itp) {
                if (itp->layerName == spec.layerName) {
                    if (itp->size.depth != spec.size.depth
                            || itp->size.x != spec.size.x
                            || itp->size.y != spec.size.y) {
                        err << "Repeated layer spec for \"" << spec.layerName
                            << "\" must have the same size" << endl;
                        throw exceptionConfigFile();
                    }
                }
            }
        }

        // Check from parameter:
        if (spec.fromLayerName.size() == 0) {
            err << "All hidden and output layers need a from parameter" << endl;
            throw exceptionConfigFile();
        }

        // Verify from layer and compute its index
        auto iti = find_if(params.begin(), params.end() - 1, [spec](topologyConfigSpec_t &pspec) {
                            return pspec.layerName == spec.fromLayerName; });
        if (iti == params.end() - 1) {
            err << "Undefined from-layer:" << spec.fromLayerName << endl;
            throw exceptionConfigFile();
        } else {
            spec.fromLayerIndex = distance(params.begin(), iti);
        }

        // If a size param was not specified, copy the size from the from-layer:
        if (!spec.sizeSpecified) {
            spec.size = params[spec.fromLayerIndex].size;
        }

        // Check that radius is not specified at the same with with a convolve or pool parameter:
        if (spec.radiusSpecified && (spec.isConvolutionFilterLayer
                || spec.isConvolutionNetworkLayer || spec.isPoolingLayer)) {
            err << "Radius cannot be specified on a convolve or pool layer." << endl;
            throw exceptionConfigFile();
        }

        // Check convolve kernel size:
        if ((spec.isConvolutionFilterLayer || spec.isConvolutionNetworkLayer)
                    && (spec.kernelSize.x == 0 || spec.kernelSize.y == 0)) {
            err << "Error in topology config file: Convolve kernel dimension cannot be zero" << endl;
            throw exceptionConfigFile();
        }
    }

    // Specific only to output layer:
    if (params.back().layerName != "output") {
        err << "Last layer must be named output" << endl;
        throw exceptionConfigFile();
    }

    if (params.back().size.depth != 1 || params.back().isConvolutionNetworkLayer) {
        err << "Output layer cannot have depth (e.g., cannot be a convolution network layer)" << endl;
        throw exceptionConfigFile();
    }
}


// Returns an array (vector) of topologyConfigSpec_t objects containing all
// the layer parameters for all the layers, extracted or derived from the
// topology config file.
// See the GitHub wiki (https://github.com/davidrmiller/neural2d)
// for more information about the format of the topology config file.
// Throws an exception for any error.
//
vector<topologyConfigSpec_t> Net::parseTopologyConfig(std::istream &cfg)
{
    if (!cfg) {
        err << "Error reading topology config stream" << endl;
        throw exceptionConfigFile();
    }

    vector<topologyConfigSpec_t> allLayers;

    unsigned lineNum = 0;
    string line;
    while (!cfg.eof() && getline(cfg, line)) {
        ++lineNum;
        if (line[0] != '\n' && line[0] != '\0' && line[0] != '#') { // this check may not be needed here any longer? !!!
            topologyConfigSpec_t params;
            params.configLineNum = lineNum;
            if (extractOneLayerParams(params, line)) { // Get what we can from the topology config file
                allLayers.push_back(params);
            }
        }
    }
    consistency(allLayers);      // Add missing fields

    return allLayers;
}

} // End namespace NNet
