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
}


void configErrorThrow(topologyConfigSpec_t &params, const string &msg)
{
    err << "There's a problem in the topology config file at line " << params.configLineNum << ":";
    if (params.layerParams.layerName.size() > 0) {
        err << "(layer \"" << params.layerParams.layerName << "\")";
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

    // Transpose the matrix so that we can access elements as [x][y]
    // This matters only if the matrix is asymmetric. While we're doing
    // this, we'll check that all rows have the same size.

    convolveMatrix_t convMat;
    unsigned firstRowSize = 0;

    firstRowSize = mat[0].size();
    if (0 != count_if(mat.begin() + 1, mat.end(), [firstRowSize](vector<float> row) {
             return row.size() != firstRowSize; })) {
        configErrorThrow(params, "Error in topology config file: inconsistent row size in convolve filter matrix spec");
    }

    for (unsigned y = 0; y < firstRowSize; ++y) {
        convMat.push_back(vector<float>());
        for (unsigned x = 0; x < mat.size(); ++x) {
            convMat.back().push_back(mat[x][y]);
        }
    }

    params.layerParams.convolveMatrix[0] = convMat;
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
    if      (stoken == "R")  params.layerParams.channel = NNet::R;
    else if (stoken == "G")  params.layerParams.channel = NNet::G;
    else if (stoken == "B")  params.layerParams.channel = NNet::B;
    else if (stoken == "BW") params.layerParams.channel = NNet::BW;
    else {
        configErrorThrow(params, "Unknown color channel");
    }

    params.colorChannelSpecified = true;
}


// Throws for any error.
//
void extractPoolMethod(topologyConfigSpec_t &params, std::istringstream &ss)
{
    string stoken;

    ss >> stoken;
    if (stoken == "max") {
        params.layerParams.poolMethod = POOL_MAX;
    } else if (stoken == "avg") {
        params.layerParams.poolMethod = POOL_AVG;
    } else {
        configErrorThrow(params, "Expected pool method \"max\" or \"avg\"");
    }
}


// Grammar:
//
// layer-name parameters
// parameters := parameter [ parameters ]
// parameter :=
//    input|output|layer-name
//    size dxy-spec
//    from layer-name
//    channel channel-spec
//    radius xy-spec
//    tf transfer-function-spec
//    convolve filter-spec
//    convolve xy-spec
//    pool max|avg radius xy-spec
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

    params.layerParams.layerName = stoken;

    // Extract the rest of the parameters:

    bool done = false;
    while (!done && !ss.eof() && ss.tellg() != -1) {
        string stoken;
        ss >> stoken;
        if (stoken.size() == 0) {
            break;
        }

        if (stoken == "size") {
            params.layerParams.size = extractDxySize(ss);
            params.sizeSpecified = true;
        } else if (stoken == "from") {
            ss >> params.fromLayerName;
        } else if (stoken == "channel") {
            extractChannel(params, ss);
            params.colorChannelSpecified = true;
        } else if (stoken == "radius") {
            params.layerParams.radius = extractXySize(ss);
            params.radiusSpecified = true;
        } else if (stoken == "tf") {
            ss >> params.layerParams.transferFunctionName;
            params.tfSpecified = true;
        } else if (stoken == "convolve") {
            // The next non-space char determines whether this is a
            // conv-filter or conv-network parameter:
            auto pos = ss.tellg();
            ss >> ctoken;
            if (ctoken == '{') {
                // Convolution filter spec    expects: {{},{}}
                params.layerParams.convolveMatrix.push_back(convolveMatrix_t());
                ss.seekg(pos);
                extractConvolveFilterMatrix(params, ss);
            } else {
                // Convolution network layer  expects: kernel size
                ss.seekg(pos);
                params.layerParams.kernelSize = extractXySize(ss);
                // Construct a matrix of the correct size, and make depth copies
                auto col = matColumn_t(params.layerParams.kernelSize.y, 0.0);  // to do: random initial values
                auto mat = convolveMatrix_t(params.layerParams.kernelSize.x, col);
                params.layerParams.convolveMatrix.assign(params.layerParams.size.depth, mat);
            }
        } else if (stoken == "pool") {
            extractPoolMethod(params, ss);
            ss >> stoken;
            if (stoken == "radius") {
                params.layerParams.poolSize = extractXySize(ss);
            } else {
                configErrorThrow(params, "Expected \"radius\" after the pooling method");
            }
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

    if (params[0].layerParams.layerName != "input") {
        err << "First layer must be named input" << endl;
        throw exceptionConfigFile();
    }

    if (params[0].fromLayerName.size() > 0) {
        warn << "Input layer cannot have a from parameter" << endl;
    }
    if (params[0].layerParams.convolveMatrix.size() > 0) {
        err << "Input layer cannot have a convolve parameter" << endl;
        throw exceptionConfigFile();
    }
    if (params[0].layerParams.poolSize.x != 0 || params[0].layerParams.poolSize.y != 0) {
        err << "Input layer cannot have a pool parameter" << endl;
        throw exceptionConfigFile();
    }
    if (params[0].layerParams.radius.x != Net::HUGE_RADIUS
                || params[0].layerParams.radius.y != Net::HUGE_RADIUS) {
        warn << "Input layer cannot have a radius parameter" << endl;
    }

    // In common to hidden layer and output layer specs:

    for (auto it = params.begin() + 1; it != params.end(); ++it) {
        auto &spec = *it;

        // Ensure that if a layer name is repeated with a size, the size must
        // match the size of the previous spec:
        if (spec.sizeSpecified) {
            for (auto itp = it - 1; itp != params.begin() - 1; --itp) {
                if (itp->layerParams.layerName == spec.layerParams.layerName) {
                    if (itp->layerParams.size.depth != spec.layerParams.size.depth
                            || itp->layerParams.size.x != spec.layerParams.size.x
                            || itp->layerParams.size.y != spec.layerParams.size.y) {
                        err << "Repeated layer spec for \"" << spec.layerParams.layerName
                            << "\" must have the same size" << endl;
                        throw exceptionConfigFile();
                    }
                }
            }
        }

        // Check from parameter:
        if (spec.fromLayerName.size() == 0) {
            warn << "All hidden and output layers need a from parameter" << endl;
        }

        spec.layerParams.resolveTransferFunctionName();

        // Verify from layer and compute its index
        auto iti = find_if(params.begin(), params.end() - 1, [spec](topologyConfigSpec_t &pspec) {
                            return pspec.layerParams.layerName == spec.fromLayerName; });
        if (iti == params.end() - 1) {
            err << "Undefined from-layer" << endl;
            throw exceptionConfigFile();
        } else {
            spec.fromLayerIndex = distance(params.begin(), iti);
        }

        // If a size param was not specified, copy the size from the from-layer:
        if (!spec.sizeSpecified) {
            spec.layerParams.size = params[spec.fromLayerIndex].layerParams.size;
        }

        // Set flags
        spec.layerParams.isConvolutionFilterLayer = spec.layerParams.convolveMatrix.size() == 1;
        spec.layerParams.isConvolutionNetworkLayer = spec.layerParams.convolveMatrix.size() > 1;
    }

    // Specific only to output layer:

    if (params.back().layerParams.layerName != "output") {
        err << "Last layer must be named output" << endl;
        throw exceptionConfigFile();
    }

    if (params.back().layerParams.size.depth != 1 || params.back().layerParams.isConvolutionNetworkLayer) {
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
