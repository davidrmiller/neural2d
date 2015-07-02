/*
unitTest.cpp for use with neural2d.
https://github.com/davidrmiller/neural2d
David R. Miller, 2015

To use these unit tests, build with the Makefile target "unitTest," then
execute in the same directory where you build neural2d.
*/

#include "neural2d.h"

using namespace std;

namespace NNet {

// Define to skip exception testing:
//#define SKIP_TEST_EXCEPTIONS

const bool StopAtFirstError = false;

unsigned numErrors = 0;

// Some temporary files for us to use:
const string topologyConfigFilename = "./topologyUnitTest.txt";
const string inputDataConfigFilename = "./inputDataUnitTest.txt";

class unitTestException : std::exception { };
extern float pixelToNetworkInputRange(unsigned val);

// Unit tests use the following macros to log information and report problems.
// The only console output from a unit test should be the result of invoking
// these macros.

#define LOG(s) (cerr << "LOG: " << s << endl)

#define ASSERT_EQ(c, v) { if (!((c)==(v))) { \
    cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), expected " \
    << (v) << ", got " << (c) << endl; \
    ++numErrors; \
    if (StopAtFirstError) throw unitTestException(); \
    } }

// ASSERT_FEQ() is for approximate floating point equality, of ~4 decimal digits precision:
#define ASSERT_FEQ(c, v) { float q = (c)/(v); if (q < 0.9999 || q > 1.0001) { \
    cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), expected " \
    << (v) << ", got " << (c) << endl; \
    ++numErrors; \
    if (StopAtFirstError) throw unitTestException(); \
    } }

#define ASSERT_NE(c, v) { if (!((c)!=(v))) { \
    cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), got unexpected " \
    << (v) << endl; \
    ++numErrors; \
    if (StopAtFirstError) throw unitTestException(); \
    } }

#define ASSERT_GE(c, v) { if (!((c)>=(v))) { \
    cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), expected >=" \
    << (v) << ", got " << (c) << endl; \
    ++numErrors; \
    if (StopAtFirstError) throw unitTestException(); \
    } }

// This macro verifies that evaluating the expression triggers the specified
// exception. We guarantee to evaluate the condition c exactly once.
// If the expected exception happens, we note it and discard it.
//
#define ASSERT_THROWS(c, e) \
{ \
    bool caught = false; \
    try { \
        (c); \
    } \
    catch (e &e_) { \
        (void)e_; \
        caught = true; \
    } \
    if (!caught) { \
        cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), did not catch expected exception " \
             << endl; \
        ++numErrors; \
        if (StopAtFirstError) throw unitTestException(); \
    } \
}


// For testing purposes, it's sometimes useful to force all the connection
// weights to a fixed value:
//
void setAllWeights(Net &myNet, float w)
{
    // For regular connections:
    for (auto &conn : myNet.connections) {
        conn.weight = w;
    }

    // The convolution kernel elements are weights too for testing purposes:
    for (auto &pLayer : myNet.layers) {
        for (auto &mat2D : pLayer->flatConvolveMatrix) {
            std::for_each(mat2D.begin(), mat2D.end(), [](float &elem) {
                elem = 1.0;
            });
        }
    }
}


// Given a layer specification object and a layer name, return a ptr to the spec with that name.
// Assumes a valid layer name.
//
topologyConfigSpec_t const *specNamed(vector<topologyConfigSpec_t> const &specs, std::string layerName)
{
    auto it = find_if(specs.begin(), specs.end(), [layerName](topologyConfigSpec_t const &spec) {
            return spec.layerName == layerName;
        });

    return &*it;
}


// Given a Net object and a layer name, return a ptr to the layer with that name.
// Assumes a valid layer name.
//
Layer const *layerNamed(Net const &myNet, std::string layerName)
{
    auto it = find_if(myNet.layers.begin(), myNet.layers.end(), [layerName](unique_ptr<Layer> const &pLayer) {
            return pLayer->layerName == layerName;
        });

    return (*it).get();
}


void unitTestConfigParsers()
{
    // Make an uninitialized Net to use in this section:
    Net myNet("", false);

    {
        LOG("Smoke test trivial config");

        string config =
            "input size 2x2\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        ASSERT_EQ(specs.size(), 2);
        ASSERT_EQ(specs[0].fromLayerName.size(), 0);
        ASSERT_EQ(specs[0].configLineNum, 1);
        ASSERT_EQ(specs[0].channel, NNet::BW);
        ASSERT_EQ(specs[0].flatConvolveMatrix.size(), 0);
        ASSERT_EQ(specs[0].isConvolutionFilterLayer, false);
        ASSERT_EQ(specs[0].isConvolutionNetworkLayer, false);
        ASSERT_EQ(specs[0].isPoolingLayer, false);
        ASSERT_EQ(specs[0].layerName, "input");
        ASSERT_EQ(specs[0].poolSize.x, 0);
        ASSERT_EQ(specs[0].poolSize.y, 0);
        ASSERT_EQ(specs[0].size.depth, 1);
        ASSERT_EQ(specs[0].size.x, 2);
        ASSERT_EQ(specs[0].size.y, 2);

        ASSERT_EQ(specs[1].fromLayerName, "input");
        ASSERT_EQ(specs[1].configLineNum, 2);
        ASSERT_EQ(specs[1].flatConvolveMatrix.size(), 0);
        ASSERT_EQ(specs[1].isConvolutionFilterLayer, false);
        ASSERT_EQ(specs[1].isConvolutionNetworkLayer, false);
        ASSERT_EQ(specs[1].isPoolingLayer, false);
        ASSERT_EQ(specs[1].layerName, "output");
        ASSERT_EQ(specs[1].poolSize.x, 0);
        ASSERT_EQ(specs[1].poolSize.y, 0);
        ASSERT_EQ(specs[1].size.depth, 1);
        ASSERT_EQ(specs[1].size.x, 1);
        ASSERT_EQ(specs[1].size.y, 1);
        ASSERT_EQ(specs[1].transferFunctionName.size(), sizeof "tanh" - 1);
    }

    {
        LOG("Test comments and blank lines in config file");

        string config =
            "#comment\n"
            "input size 2x2\n"
            " #comment\n"
            "\n"
            "output size 1 from input\n"
            "\n"
            "#\n"
            " #\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        ASSERT_EQ(specs.size(), 2);
    }

    {
        LOG("Input channel parameter");

        string config =
            "input size 2x2 channel R\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs[0].channel, NNet::R);
    }

    {
        LOG("Test dxySize: only X given");

        string config =
            "input size 3\n"
            "output from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs[0].size.depth, 1);
        ASSERT_EQ(specs[0].size.x, 3);
        ASSERT_EQ(specs[0].size.y, 1);
    }

    {
        LOG("Test dxySize(): depth and X given");

        string config =
            "input size 1\n"
            "layerHidden size 4*3 from input convolve 1x1\n" // missing y
            "output size 1 from layerHidden\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs[1].size.depth, 4);
        ASSERT_EQ(specs[1].size.x, 3);
        ASSERT_EQ(specs[1].size.y, 1);
    }

    {
        LOG("Test dxySize: depth, X, and Y given");

        string config =
            "input size 1\n"
            "layerHidden size 4*3x5 from input convolve 1x1\n"
            "output size 1 from layerHidden\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs[1].size.depth, 4);
        ASSERT_EQ(specs[1].size.x, 3);
        ASSERT_EQ(specs[1].size.y, 5);
    }

    {
        LOG("Test whitespace tolerance");

        string config =
            "  input size 1\n"
            "output from input \n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs.size(), 2);
        ASSERT_EQ(specs[0].layerName, "input");
    }

    {
        LOG("Test whitespace tolerance 2");

        string config =
            "input\tsize 5x6\t\n"
            "\toutput size 2\tfrom input\n\t";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs.size(), 2);
        ASSERT_EQ(specs[0].size.x, 5);
        ASSERT_EQ(specs[1].size.x, 2);
        ASSERT_EQ(specs[1].size.y, 1);
    }

    {
        LOG("from parameter");

        string config = \
            "input size 1\n" \
            "layer1 size 1 from input\n" \
            "layer2 size 2x2 from layer1 \n" \
            "layer3 size 7x8 from input\n" \
            "layer4 size 2x2 from layer3\n" \
            "layer5 from input\n" \
            "layer6 from layer4\n" \
            "output size 1 from layer6\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs.size(), 8);

        auto const *spec = &specs[0];
        ASSERT_EQ(spec->fromLayerName.size(), 0);
        ASSERT_EQ(spec->configLineNum, 1);
        ASSERT_EQ(spec->channel, NNet::BW);
        ASSERT_EQ(spec->flatConvolveMatrix.size(), 0);
        ASSERT_EQ(spec->isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);
        ASSERT_EQ(spec->layerName, "input");
        ASSERT_EQ(spec->poolSize.x, 0);
        ASSERT_EQ(spec->poolSize.y, 0);
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 1);
        ASSERT_EQ(spec->size.y, 1);

        spec = specNamed(specs, "layer1");
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "input" - 1);
        ASSERT_EQ(spec->configLineNum, 2);
        ASSERT_EQ(spec->channel, NNet::BW);
        ASSERT_EQ(spec->flatConvolveMatrix.size(), 0);
        ASSERT_EQ(spec->isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);
        ASSERT_EQ(spec->layerName, "layer1");
        ASSERT_EQ(spec->poolSize.x, 0);
        ASSERT_EQ(spec->poolSize.y, 0);
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 1);
        ASSERT_EQ(spec->size.y, 1);
        ASSERT_EQ(spec->transferFunctionName.size(), sizeof "tanh" - 1);

        spec = specNamed(specs, "layer2");
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "layer1" - 1);
        ASSERT_EQ(spec->configLineNum, 3);
        ASSERT_EQ(spec->channel, NNet::BW);
        ASSERT_EQ(spec->flatConvolveMatrix.size(), 0);
        ASSERT_EQ(spec->isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);
        ASSERT_EQ(spec->layerName, "layer2");
        ASSERT_EQ(spec->poolSize.x, 0);
        ASSERT_EQ(spec->poolSize.y, 0);
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 2);
        ASSERT_EQ(spec->size.y, 2);
        ASSERT_EQ(spec->transferFunctionName.size(), sizeof "tanh" - 1);

        spec = specNamed(specs, "layer3");
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "input" - 1);
        ASSERT_EQ(spec->configLineNum, 4);
        ASSERT_EQ(spec->channel, NNet::BW);
        ASSERT_EQ(spec->flatConvolveMatrix.size(), 0);
        ASSERT_EQ(spec->isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);
        ASSERT_EQ(spec->layerName, "layer3");
        ASSERT_EQ(spec->poolSize.x, 0);
        ASSERT_EQ(spec->poolSize.y, 0);
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 7);
        ASSERT_EQ(spec->size.y, 8);
        ASSERT_EQ(spec->transferFunctionName.size(), sizeof "tanh" - 1);

        spec = specNamed(specs, "layer4");
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "layer3" - 1);
        ASSERT_EQ(spec->configLineNum, 5);
        ASSERT_EQ(spec->channel, NNet::BW);
        ASSERT_EQ(spec->flatConvolveMatrix.size(), 0);
        ASSERT_EQ(spec->isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);
        ASSERT_EQ(spec->layerName, "layer4");
        ASSERT_EQ(spec->poolSize.x, 0);
        ASSERT_EQ(spec->poolSize.y, 0);
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 2);
        ASSERT_EQ(spec->size.y, 2);
        ASSERT_EQ(spec->transferFunctionName.size(), sizeof "tanh" - 1);
    }

    {
        LOG("radius parameter");

        string config =
            "input size 1\n"
            "layer1 size 1 from input radius 2x3\n"
            "layer2 size 1 from layer1 radius 4\n"
            "layer3 size 1 from layer2 radius 0x4\n"
            "output size 1 from layer3\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->radius.x, 2);
        ASSERT_EQ(spec->radius.y, 3);

        spec = &specs[2];
        ASSERT_EQ(spec->radius.x, 4);
        ASSERT_EQ(spec->radius.y, 1);

        spec = &specs[3];
        ASSERT_EQ(spec->radius.x, 0);
        ASSERT_EQ(spec->radius.y, 4);

        spec = &specs[4];
        ASSERT_GE(spec->radius.x, 0.0);
        ASSERT_GE(spec->radius.y, 0.0);
    }

    {
        LOG("tf parameter");

        string config =
            "input size 1\n"
            "layer1 size 1 from input tf linear\n"
            "layer2 size 1 from layer1 radius 4 tf gaussian\n"
            "layer3 size 1 from layer2 radius 0x4\n"
            "output size 1 from layer3 tf logistic\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->transferFunctionName, "linear");
        spec = &specs[2];
        ASSERT_EQ(spec->transferFunctionName, "gaussian");
        spec = &specs[3];
        ASSERT_EQ(spec->transferFunctionName, "tanh");
        spec = &specs[4];
        ASSERT_EQ(spec->transferFunctionName, "logistic");
    }

    {
        LOG("test that missing size matches from-layer");

        string config =
            "input size 2x3\n"
            "layer1 from input\n"
            "layer2 size 4x5 from input\n"
            "layer3 from layer2\n"
            "layer4 size 2*3x4 from input convolve 1x1\n"
            "layer5 from layer4 convolve 1x1\n"
            "output size 1 from layer5\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto const *spec = specNamed(specs, "layer1");
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 2);
        ASSERT_EQ(spec->size.y, 3);

        spec = specNamed(specs, "layer3");
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 4);
        ASSERT_EQ(spec->size.y, 5);

        spec = specNamed(specs, "layer5");
        ASSERT_EQ(spec->size.depth, 2);
        ASSERT_EQ(spec->size.x, 3);
        ASSERT_EQ(spec->size.y, 4);

        ASSERT_EQ(specs.back().size.depth, 1);
        ASSERT_EQ(specs.back().size.x, 1);
        ASSERT_EQ(specs.back().size.y, 1);
    }

    {
        LOG("test that missing size matches from-layer 2");

        string config =
            "input size 1\n"
            "output from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->size.depth, 1);
        ASSERT_EQ(spec->size.x, 1);
        ASSERT_EQ(spec->size.y, 1);
    }

    {
        LOG("convolve filter matrix spec");

        string config =
            "input size 16x16\n"
            "layer1 from input convolve {2}\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->isConvolutionFilterLayer, true);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);
        ASSERT_EQ(spec->flatConvolveMatrix.size(), 1);    // depth = 1
        ASSERT_EQ(spec->flatConvolveMatrix[0].size(), 1); // one kernel element
    }

    {
        LOG("convolve filter matrix spec 2");

        string config =
            "input size 16x16\n"
            "layer1 from input convolve {2,3}\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->isConvolutionFilterLayer, true);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);

        ASSERT_EQ(spec->flatConvolveMatrix.size(), 1);
        ASSERT_EQ(spec->flatConvolveMatrix[0].size(), 2*1);
        ASSERT_EQ(spec->flatConvolveMatrix[0][0], 2);
        ASSERT_EQ(spec->flatConvolveMatrix[0][1], 3);
    }

    {
        LOG("convolve filter matrix spec 3");

        string config =
            "input size 16x16\n"
            "layer1 from input convolve {{2},{3}}\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->isConvolutionFilterLayer, true);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);

        ASSERT_EQ(spec->flatConvolveMatrix.size(), 1);
        ASSERT_EQ(spec->flatConvolveMatrix[0].size(), 1*2);
        ASSERT_EQ(spec->flatConvolveMatrix[0][0], 2);
        ASSERT_EQ(spec->flatConvolveMatrix[0][1], 3);
    }

    {
        LOG("convolve filter matrix spec oriention");

        string config =
            "input size 16x16\n"
            "layer1 from input convolve {{1,2,3},{4,5,6}, {7,8,9}}\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->isConvolutionFilterLayer, true);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->isPoolingLayer, false);

        ASSERT_EQ(spec->flatConvolveMatrix.size(), 1);
        ASSERT_EQ(spec->flatConvolveMatrix[0].size(), 3*3);
        ASSERT_EQ(spec->flatConvolveMatrix[0][flattenXY(0, 0, 3)], 1);
        ASSERT_EQ(spec->flatConvolveMatrix[0][flattenXY(1, 0, 3)], 2);
        ASSERT_EQ(spec->flatConvolveMatrix[0][flattenXY(0, 1, 3)], 4);
        ASSERT_EQ(spec->flatConvolveMatrix[0][flattenXY(2, 2, 3)], 9);
    }

    {
        LOG("convolve network kernel size param");

        string config =
            "input size 16x16\n"
            "layer1 size 10*16x16 from input convolve 3x4\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->size.depth, 10);
        ASSERT_EQ(spec->size.x, 16);
        ASSERT_EQ(spec->size.y, 16);
        ASSERT_EQ(spec->kernelSize.x, 3);
        ASSERT_EQ(spec->kernelSize.y, 4);
        ASSERT_EQ(spec->isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->isConvolutionNetworkLayer, true);
        ASSERT_EQ(spec->isPoolingLayer, false);

        ASSERT_EQ(spec->flatConvolveMatrix.size(), 10);
        ASSERT_EQ(spec->flatConvolveMatrix[0].size(), 3*4);
    }

#ifndef SKIP_TEST_EXCEPTIONS
    {
        LOG("convolve filter matrix spec unequal rows");

        string config =
            "input size 16x16\n"
            "layer1 from input convolve {{2},{3,2}}\n"
            "output from input\n";

        istringstream ss(config);

        ASSERT_THROWS(myNet.parseTopologyConfig(ss), exceptionConfigFile);
    }
#endif

    {
        LOG("pool param");

        string config =
            "input size 16x16\n"
            "layer1 size 10*16x16 from input pool max 2x3\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->size.depth, 10);
        ASSERT_EQ(spec->poolMethod, POOL_MAX);
        ASSERT_EQ(spec->poolSize.x, 2);
        ASSERT_EQ(spec->poolSize.y, 3);
        ASSERT_EQ(spec->isPoolingLayer, true);
    }

    {
        LOG("convolve networking param");

        string config =
            "input size 16x16\n"
            "layer1 size 10*16x16 from input convolve 3x5\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        auto *spec = &specs[1];
        ASSERT_EQ(spec->kernelSize.x, 3);
        ASSERT_EQ(spec->kernelSize.y, 5);
        ASSERT_EQ(spec->size.depth, 10);

        ASSERT_EQ(spec->flatConvolveMatrix.size(), 10);
        ASSERT_EQ(spec->flatConvolveMatrix[0].size(), 3*5);
    }

    {
        LOG("layer spec reordering");

        // The reordering should put the input spec first, and the output spec(s) last,
        // and reorder the hidden layers by their from parameter.

        string config =
            "output from layer3\n"
            "layer1 from input\n"
            "output from layer5\n"
            "output from layer2\n"
            "output from layer4\n"
            "input size 1\n"
            "layer3 from layer2\n"
            "output from layer1\n"
            "layer2 from layer1\n"
            "layer5 from layer4\n"
            "layer4 from layer3\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);

        ASSERT_EQ(specs.size(), 11);  // in, out, 5 hidden, plus 4 extra output specs
        ASSERT_EQ(specs[0].layerName, "input");
        ASSERT_EQ(specs[1].layerName, "layer1");
        ASSERT_EQ(specs[2].layerName, "layer2");
        ASSERT_EQ(specs[3].layerName, "layer3");
        ASSERT_EQ(specs[4].layerName, "layer4");
        ASSERT_EQ(specs[5].layerName, "layer5");
        ASSERT_EQ(specs[6].layerName, "output");
        ASSERT_EQ(specs[7].layerName, "output");
        ASSERT_EQ(specs[8].layerName, "output");
        ASSERT_EQ(specs[9].layerName, "output");
        ASSERT_EQ(specs[10].layerName, "output");
    }
}


void unitTestNet()
{
    LOG("unitTestNet()");

    {
        LOG("Smoke test null config");
        Net myNet("", false);
    }

    {
        LOG("Smoke test trivial config");
        string config =
            "input size 1\n"
            "output from input\n";

        istringstream ss(config);
        Net myNet("", false);
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_FEQ(myNet.alpha, 0.1f);
        ASSERT_EQ(myNet.bias.output, 1.0);
        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 1);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 1);

        const Neuron *pNeuron = &myNet.layers[0]->neurons[0][0];
        ASSERT_EQ(pNeuron->forwardConnectionsIndices.size(), 1);
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 0);

        pNeuron = &myNet.layers[1]->neurons[0][0];
        ASSERT_EQ(pNeuron->forwardConnectionsIndices.size(), 0);
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 2);

        ASSERT_EQ(myNet.connections.size(), 2);
    }

    {
        LOG("Neurons and connections");
        string config =
            "input size 10x10\n"
            "output size 8x6 from input\n";

        istringstream ss(config);
        Net myNet("", false);
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 10*10);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 8*6);

        const Neuron *pNeuron = &myNet.layers[0]->neurons[0][0];
        ASSERT_EQ(pNeuron->forwardConnectionsIndices.size(), 8*6);
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 0);

        ASSERT_EQ(myNet.connections.size(), 8*6*10*10 + 8*6);
    }

    {
        LOG("neuron connections");

        string topologyConfig =
            "input size 1\n"
            "output size 1 from input radius 0x0 tf linear\n"; // One-to-one

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        Net myNet(topologyConfigFilename, false);

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 1);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 1);

        ASSERT_EQ(myNet.layers[0]->isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[0]->isConvolutionNetworkLayer, false);
        ASSERT_EQ(myNet.layers[0]->isPoolingLayer, false);
        ASSERT_EQ(myNet.layers[1]->isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[1]->isConvolutionNetworkLayer, false);
        ASSERT_EQ(myNet.layers[1]->isPoolingLayer, false);

        ASSERT_EQ(myNet.layers[0]->size.depth, 1);
        ASSERT_EQ(myNet.layers[0]->size.x, 1);
        ASSERT_EQ(myNet.layers[0]->size.y, 1);

        ASSERT_EQ(myNet.layers[1]->size.depth, 1);
        ASSERT_EQ(myNet.layers[1]->size.x, 1);
        ASSERT_EQ(myNet.layers[1]->size.y, 1);

        ASSERT_EQ(myNet.connections.size(), 1+1); // one plus bias
        ASSERT_EQ(myNet.layers[0]->neurons[0][0].backConnectionsIndices.size(), 0);
        ASSERT_EQ(myNet.layers[0]->neurons[0][0].forwardConnectionsIndices.size(), 1);
        ASSERT_EQ(myNet.layers[1]->neurons[0][0].backConnectionsIndices.size(), 2); // Source plus bias
        ASSERT_EQ(myNet.layers[1]->neurons[0][0].forwardConnectionsIndices.size(), 0);

        ASSERT_EQ(myNet.bias.backConnectionsIndices.size(), 0);
        ASSERT_EQ(myNet.bias.forwardConnectionsIndices.size(), 1);

        auto const &l0n0 = myNet.layers[0]->neurons[0][0];
        auto const &l1n0 = myNet.layers[1]->neurons[0][0];
        auto backIdx = l1n0.backConnectionsIndices[0];
        auto forwardIdx = l0n0.forwardConnectionsIndices[0];
        ASSERT_EQ(backIdx, forwardIdx); // Should refer to the same connection record

        auto const &conn = myNet.connections[backIdx];
        ASSERT_EQ(&conn.fromNeuron, &l0n0);
        ASSERT_EQ(&conn.toNeuron, &l1n0);

        ASSERT_EQ(&myNet.connections[myNet.bias.forwardConnectionsIndices[0]].toNeuron, &l1n0);
        ASSERT_EQ(&myNet.connections[l1n0.backConnectionsIndices[1]].fromNeuron, &myNet.bias);
    }

    {
        LOG("neuron layer construction and depth");

        string topologyConfig =
            "input size 8x8 channel G\n"
            "output size 8x8 from input radius 0x1 tf linear\n"; // One col, 3 rows

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        Net myNet(topologyConfigFilename, false);

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 8*8);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 8*8);

        ASSERT_EQ(myNet.layers[0]->isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[0]->isConvolutionNetworkLayer, false);
        ASSERT_EQ(myNet.layers[0]->isPoolingLayer, false);
        ASSERT_EQ(myNet.layers[1]->isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[1]->isConvolutionNetworkLayer, false);
        ASSERT_EQ(myNet.layers[1]->isPoolingLayer, false);

        ASSERT_EQ(myNet.layers[0]->size.depth, 1);
        ASSERT_EQ(myNet.layers[0]->size.x, 8);
        ASSERT_EQ(myNet.layers[0]->size.y, 8);

        ASSERT_EQ(myNet.layers[1]->size.depth, 1);
        ASSERT_EQ(myNet.layers[1]->size.x, 8);
        ASSERT_EQ(myNet.layers[1]->size.y, 8);
    }

    {
        LOG("Layer depth compatiblity");

        // This tests all 64 ways of connecting pairs of these four kinds of layers:
        //     regular, convolution filtering, convolution networking, pooling
        // in these two depths:
        //     depth = 1, and depth = 2

        string topologyConfig =
            "input size 3x4\n"
            "layer1 from input radius 0x0 tf linear\n"                // regular depth 1  to regular depth 1
            "layer2 size 2*3x4 from layer1 radius 0x0 tf linear\n"    // regular depth 1  to regular depth >1
            "layer3 size 3x4 from layer2 radius 0x0 tf linear\n"      // regular depth >1 to regular depth 1
            "layer4 size 1*3x4 from layer3 convolve {1}\n"            // regular depth 1  to conv-filter depth 1
            "layer5 from layer4 convolve {1}\n"                       // conv-filter depth 1  to conv-filter depth 1
            "layer6 size 3x4 from layer5 radius 0x0 tf linear\n"      // conv-filter depth 1  to regular depth 1
            "layer7 size 2*3x4 from layer6 convolve {1}\n"            // regular depth 1  to conv-filter depth >1
            "layer8 size 1*3x4 from layer7 convolve {1}\n"            // conv-filter depth >1 to conv-filter depth 1
            "layer9 size 2*3x4 from layer8 convolve {1}\n"            // conv-filter depth 1  to conv-filter depth >1
            "layer10 size 3x4 from layer9 radius 0x0 tf linear\n"     // conv-filter depth >1 to regular depth 1
            "layer11 from layer10 convolve 1x1\n"                     // regular depth 1  to conv-net depth 1
            "layer12 from layer11 convolve 1x1\n"                     // conv-net depth 1  to conv-net depth 1
            "layer13 size 2*3x4 from layer12 convolve 1x1\n"          // conv-net depth 1  to conv-net depth >1
            "layer14 size 1*3x4 from layer13 convolve 1x1\n"          // conv-net depth >1 to conv-net depth 1
            "layer15 size 3x4 from layer14 radius 0x0 tf linear\n"    // conv-net depth 1  to regular depth 1
            "layer16 size 2*3x4 from layer15 convolve 1x1\n"          // regular depth 1  to conv-net depth >1
            "layer17 from layer16 convolve 1x1\n"                     // conv-net depth >1 to conv-net depth >1
            "layer18 size 3x4 from layer17 radius 0x0 tf linear\n"    // conv-net depth >1 to regular depth 1
            "layer19 size 1*3x4 from layer18 pool max 1x1\n"          // regular depth 1  to pooling depth 1
            "layer20 from layer19 pool max 1x1\n"                     // pooling depth 1  to pooling depth 1
            "layer21 size 2*3x4 from layer20 pool max 1x1\n"          // pooling depth 1  to pooling depth >1
            "layer22 size 1*3x4 from layer21 pool max 1x1\n"          // pooling depth >1 to pooling depth 1
            "layer23 size 3x4 from layer22 radius 0x0 tf linear\n"    // pooling depth 1  to regular depth 1
            "layer24 size 2*3x4 from layer23 pool max 1x1\n"          // regular depth 1  to pooling depth >1
            "layer25 from layer24 pool max 1x1\n"                     // pooling depth >1 to pooling depth >1
            "layer26 size 3x4 from layer25 radius 0x0 tf linear\n"    // pooling depth >1 to regular depth 1
            "layer27 size 2*3x4 from layer26 radius 0x0 tf linear\n"  // already checked before
            "layer28 from layer27 radius 0x0 tf linear\n"             // regular depth >1 to regular depth >1
            "layer29 size 1*3x4 from layer28 convolve {1}\n"          // regular depth >1 to conv-filter depth 1
            "layer30 size 2*3x4 from layer29 radius 0x0 tf linear\n"  // conv-filter depth 1  to regular depth >1
            "layer31 from layer30 convolve {1}\n"                     // regular depth >1 to conv-filter depth >1
            "layer32 from layer31 convolve {1}\n"                     // conv-filter depth >1 to conv-filter depth >1
            "layer33 from layer32 radius 0x0 tf linear\n"             // conv-filter depth >1 to regular depth >1
            "layer34 size 1*3x4 from layer33 convolve 1x1\n"          // regular depth >1 to conv-net depth 1
            "layer35 size 2*3x4 from layer34 radius 0x0 tf linear\n"  // conv-net depth 1  to regular depth >1
            "layer36 from layer35 convolve 1x1\n"                     // regular depth >1 to conv-net depth >1
            "layer37 size 1*3x4 from layer36 convolve {1}\n"          // conv-net depth >1 to conv-filter depth 1
            "layer38 from layer37 convolve 1x1\n"                     // conv-filter depth  1 to conv-net depth 1
            "layer39 from layer38 convolve {1}\n"                     // conv-net depth  1 to conv-filter depth 1
            "layer40 size 2*3x4 from layer39 convolve 1x1\n"          // conv-filter depth  1 to conv-net depth >1
            "layer41 from layer40 convolve {1}\n"                     // conv-net depth >1 to conv-filter depth >1
            "layer42 size 1*3x4 from layer41 convolve 1x1\n"          // conv-filter depth >1 to conv-net depth 1
            "layer43 size 2*3x4 from layer42 convolve {1}\n"          // conv-net depth  1 to conv-filter depth >1
            "layer44 size 1*3x4 from layer43 pool max 1x1\n"          // conv-filter depth >1 to pooling depth 1
            "layer45 size 2*3x4 from layer44 radius 0x0 tf linear\n"  // pooling depth 1  to regular depth >1
            "layer46 size 1*3x4 from layer45 pool max 1x1\n"          // regular depth >1 to pooling depth 1
            "layer47 from layer46 convolve {1}\n"                     // pooling depth 1  to conv-filter depth 1
            "layer48 from layer47 pool max 1x1\n"                     // conv-filter depth  1 to pooling depth 1
            "layer49 size 2*3x4 from layer48 convolve {1}\n"          // pooling depth 1  to conv-filter depth >1
            "layer50 from layer49 convolve 1x1\n"                     // conv-filter depth >1 to conv-net depth >1
            "layer51 from layer50 radius 0x0 tf linear\n"             // conv-net depth >1 to regular depth >1
            "layer52 from layer51 pool max 1x1\n"                     // regular depth >1 to pooling depth >1
            "layer53 from layer52 radius 0x0 tf linear\n"             // pooling depth >1 to regular depth >1
            "layer54 from layer53 pool max 1x1\n"                     // already checked before
            "layer55 size 1*3x4 from layer54 convolve {1}\n"          // pooling depth >1 to conv-filter depth 1
            "layer56 size 2*3x4 from layer55 pool max 1x1\n"          // conv-filter depth  1 to pooling depth >1
            "layer57 size 1*3x4 from layer56 convolve 1x1\n"          // pooling depth >1 to conv-net depth 1
            "layer58 from layer57 pool max 1x1\n"                     // conv-net depth  1 to pooling depth 1
            "layer59 from layer58 convolve 1x1\n"                     // pooling depth 1  to conv-net depth 1
            "layer60 size 2*3x4 from layer59 pool max 1x1\n"          // conv-net depth  1 to pooling depth >1
            "layer61 from layer60 convolve 1x1\n"                     // pooling depth >1 to conv-net depth >1
            "layer62 size 1*3x4 from layer61 pool max 1x1\n"          // conv-net depth >1 to pooling depth 1
            "layer63 size 2*3x4 from layer62 convolve 1x1\n"          // pooling depth 1  to conv-net depth >1
            "layer64 from layer63 pool max 1x1\n"                     // conv-net depth >1 to pooling depth >1
            "layer65 from layer64 convolve {1}\n"                     // pooling depth >1 to conv-filter depth >1
            "layer66 from layer65 pool max 1x1\n"                     // conv-filter depth >1 to pooling depth >1
            "output size 3x4 from layer66 radius 0x0 tf linear\n";

        string inputDataConfig =
            "{ 0.125 0.25 0.5 0.75 } \n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        myNet.debugShowNet();
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        ASSERT_FEQ(myNet.layers[0]->neurons[0][flattenXY(0,0,2)].output, 0.125f);

        for (auto const &pLayer : myNet.layers) {
            ASSERT_EQ(pLayer->size.x, 3);
            ASSERT_EQ(pLayer->size.y, 4);
            ASSERT_EQ(pLayer->neurons[0].size(), 3*4);
        }

        // We expect the following layers to have depth == 1:
        vector<uint32_t> hasDepth1 { 0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 18, 19, 20,
                  22, 23, 26, 29, 34, 37, 38, 39, 42, 44, 46, 47, 48, 55, 57, 58, 59, 62, 67 };

        // We expect the following layers to have depth == 2:
        vector<uint32_t> hasDepth2 { 2, 7, 9, 13, 16, 17, 21, 24, 25, 27, 28, 30, 31,
                  32, 33, 35, 36, 40, 41, 43, 45, 49, 50, 51, 52, 53, 54, 56, 60, 61, 63, 64, 65, 66 };

        for (uint32_t layerNum : hasDepth1) {
            ASSERT_EQ(myNet.layers[layerNum]->neurons.size(), 1);
        }

        for (uint32_t layerNum : hasDepth2) {
            ASSERT_EQ(myNet.layers[layerNum]->neurons.size(), 2);
        }

        // We expect the following layers to have a one-to-one connection with a single
        // plane of a XxY patch of source neurons (equals depth*X*Y connections) and no bias:
        vector<uint32_t> singlePlaneOneToOneNoBias { 4, 5, 7, 9, 11, 12, 13, 16, 17, 19, 20, 21, 24, 25,
                  31, 32, 36, 38, 39, 40, 41, 43, 47, 48, 49, 50, 52, 56, 58, 59, 60, 61, 63, 64, 65, 66 };

        uint32_t expectedSize = 1*1; // patch size
        for (uint32_t layerNum : singlePlaneOneToOneNoBias) {
            Neuron const &exemplaryNeuron = myNet.layers[layerNum]->neurons[0][0];
            ASSERT_EQ(exemplaryNeuron.backConnectionsIndices.size(), expectedSize);
        }

        // We expect the following layers to have a one-to-one connection with a single
        // plane of a XxY patch of source neurons (equals depth*X*Y connections) plus a bias:
        vector<uint32_t> singlePlaneOneToOnePlusBias { 2, 6, 15, 23, 28, 30, 33, 35, 45, 51, 53 };

        expectedSize = 1*1 + 1; // patch size plus a bias
        for (uint32_t layerNum : singlePlaneOneToOnePlusBias) {
            Neuron const &exemplaryNeuron = myNet.layers[layerNum]->neurons[0][0];
            ASSERT_EQ(exemplaryNeuron.backConnectionsIndices.size(), expectedSize);
        }

        // We expect the following layers to have a one-to-one connection with depth planes
        // of a XxY patch of source neurons (equals depth*X*Y connections) and no bias:
        vector<uint32_t> multiplePlaneOneToOneNoBias { 8, 14, 22, 29, 34, 37, 42, 44, 46, 55, 57, 62 };

        expectedSize = 2*1*1; // depth * patch size, no bias
        for (uint32_t layerNum : multiplePlaneOneToOneNoBias) {
            Neuron const &exemplaryNeuron = myNet.layers[layerNum]->neurons[0][0];
            ASSERT_EQ(exemplaryNeuron.backConnectionsIndices.size(), expectedSize);
        }

        // We expect the following layers to have a one-to-one connection with depth planes
        // of a XxY patch of source neurons (equals depth*X*Y connections) plus a bias:
        vector<uint32_t> multiplePlaneOneToOneWithBias { 3, 10, 18, 26 };

        expectedSize = 2*1*1 + 1; // depth * patch size, plus a bias
        for (uint32_t layerNum : multiplePlaneOneToOneWithBias) {
            Neuron const &exemplaryNeuron = myNet.layers[layerNum]->neurons[0][0];
            ASSERT_EQ(exemplaryNeuron.backConnectionsIndices.size(), expectedSize);
        }

        // Follow the top left neuron output:
        float val = 0.125;
        ASSERT_FEQ(myNet.layers[ 0]->neurons[0][0].output, val = 0.125f); // input size 3x4
        ASSERT_FEQ(myNet.layers[ 1]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer1 from input radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[ 2]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer2 size 2*3x4 from layer1 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[ 3]->neurons[0][0].output, val = 2.0f * val + 1.0f); // layer3 size 3x4 from layer2 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[ 4]->neurons[0][0].output, val = 1.0f * val       ); // layer4 size 1*3x4 from layer3 convolve {1}
        ASSERT_FEQ(myNet.layers[ 5]->neurons[0][0].output, val = 1.0f * val       ); // layer5 from layer4 convolve {1}
        ASSERT_FEQ(myNet.layers[ 6]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer6 size 3x4 from layer5 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[ 7]->neurons[0][0].output, val = 1.0f * val       ); // layer7 size 2*3x4 from layer6 convolve {1}
        ASSERT_FEQ(myNet.layers[ 8]->neurons[0][0].output, val = 2.0f * val       ); // layer8 size 1*3x4 from layer7 convolve {1}
        ASSERT_FEQ(myNet.layers[ 9]->neurons[0][0].output, val = 1.0f * val       ); // layer9 size 2*3x4 from layer8 convolve {1}
        ASSERT_FEQ(myNet.layers[10]->neurons[0][0].output, val = 2.0f * val + 1.0f); // layer10 size 3x4 from layer9 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[11]->neurons[0][0].output, val = 1.0f * val       ); // layer11 from layer10 convolve 1x1
        ASSERT_FEQ(myNet.layers[12]->neurons[0][0].output, val = 1.0f * val       ); // layer12 from layer11 convolve 1x1
        ASSERT_FEQ(myNet.layers[13]->neurons[0][0].output, val = 1.0f * val       ); // layer13 size 2*3x4 from layer12 convolve 1x1
        ASSERT_FEQ(myNet.layers[14]->neurons[0][0].output, val = 2.0f * val       ); // layer14 size 1*3x4 from layer13 convolve 1x1
        ASSERT_FEQ(myNet.layers[15]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer15 size 3x4 from layer14 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[16]->neurons[0][0].output, val = 1.0f * val       ); // layer16 size 2*3x4 from layer15 convolve 1x1
        ASSERT_FEQ(myNet.layers[17]->neurons[0][0].output, val = 1.0f * val       ); // layer17 from layer16 convolve 1x1
        ASSERT_FEQ(myNet.layers[18]->neurons[0][0].output, val = 2.0f * val + 1.0f); // layer18 size 3x4 from layer17 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[19]->neurons[0][0].output, val =        val       ); // layer19 size 1*3x4 from layer18 pool max 1x1
        ASSERT_FEQ(myNet.layers[20]->neurons[0][0].output, val =        val       ); // layer20 from layer19 pool max 1x1
        ASSERT_FEQ(myNet.layers[21]->neurons[0][0].output, val =        val       ); // layer21 size 2*3x4 from layer20 pool max 1x1
        ASSERT_FEQ(myNet.layers[22]->neurons[0][0].output, val =        val       ); // layer22 size 1*3x4 from layer21 pool max 1x1
        ASSERT_FEQ(myNet.layers[23]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer23 size 3x4 from layer22 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[24]->neurons[0][0].output, val =        val       ); // layer24 size 2*3x4 from layer23 pool max 1x1
        ASSERT_FEQ(myNet.layers[25]->neurons[0][0].output, val =        val       ); // layer25 from layer24 pool max 1x1
        ASSERT_FEQ(myNet.layers[26]->neurons[0][0].output, val = 2.0f * val + 1.0f); // layer26 size 3x4 from layer25 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[27]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer27 size 2*3x4 from layer26 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[28]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer28 from layer27 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[29]->neurons[0][0].output, val = 2.0f * val       ); // layer29 size 1*3x4 from layer28 convolve {1}
        ASSERT_FEQ(myNet.layers[30]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer30 size 2*3x4 from layer29 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[31]->neurons[0][0].output, val = 1.0f * val       ); // layer31 from layer30 convolve {1}
        ASSERT_FEQ(myNet.layers[32]->neurons[0][0].output, val = 1.0f * val       ); // layer32 from layer31 convolve {1}
        ASSERT_FEQ(myNet.layers[33]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer33 from layer32 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[34]->neurons[0][0].output, val = 2.0f * val       ); // layer34 size 1*3x4 from layer33 convolve 1x1
        ASSERT_FEQ(myNet.layers[35]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer35 size 2*3x4 from layer34 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[36]->neurons[0][0].output, val = 1.0f * val       ); // layer36 from layer35 convolve 1x1
        ASSERT_FEQ(myNet.layers[37]->neurons[0][0].output, val = 2.0f * val       ); // layer37 size 1*3x4 from layer36 convolve {1}
        ASSERT_FEQ(myNet.layers[38]->neurons[0][0].output, val = 1.0f * val       ); // layer38 from layer37 convolve 1x1
        ASSERT_FEQ(myNet.layers[39]->neurons[0][0].output, val = 1.0f * val       ); // layer39 from layer38 convolve {1}
        ASSERT_FEQ(myNet.layers[40]->neurons[0][0].output, val = 1.0f * val       ); // layer40 size 2*3x4 from layer39 convolve 1x1
        ASSERT_FEQ(myNet.layers[41]->neurons[0][0].output, val = 1.0f * val       ); // layer41 from layer40 convolve {1}
        ASSERT_FEQ(myNet.layers[42]->neurons[0][0].output, val = 2.0f * val       ); // layer42 size 1*3x4 from layer41 convolve 1x1
        ASSERT_FEQ(myNet.layers[43]->neurons[0][0].output, val = 1.0f * val       ); // layer43 size 2*3x4 from layer42 convolve {1}
        ASSERT_FEQ(myNet.layers[44]->neurons[0][0].output, val =        val       ); // layer44 size 1*3x4 from layer43 pool max 1x1
        ASSERT_FEQ(myNet.layers[45]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer45 size 2*3x4 from layer44 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[46]->neurons[0][0].output, val =        val       ); // layer46 size 1*3x4 from layer45 pool max 1x1
        ASSERT_FEQ(myNet.layers[47]->neurons[0][0].output, val = 1.0f * val       ); // layer47 from layer46 convolve {1}
        ASSERT_FEQ(myNet.layers[48]->neurons[0][0].output, val =        val       ); // layer48 from layer47 pool max 1x1
        ASSERT_FEQ(myNet.layers[49]->neurons[0][0].output, val = 1.0f * val       ); // layer49 size 2*3x4 from layer48 convolve {1}
        ASSERT_FEQ(myNet.layers[50]->neurons[0][0].output, val = 1.0f * val       ); // layer50 from layer49 convolve 1x1
        ASSERT_FEQ(myNet.layers[51]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer51 from layer50 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[52]->neurons[0][0].output, val =        val       ); // layer52 from layer51 pool max 1x1
        ASSERT_FEQ(myNet.layers[53]->neurons[0][0].output, val = 1.0f * val + 1.0f); // layer53 from layer52 radius 0x0 tf linear
        ASSERT_FEQ(myNet.layers[54]->neurons[0][0].output, val =        val       ); // layer54 from layer53 pool max 1x1
        ASSERT_FEQ(myNet.layers[55]->neurons[0][0].output, val = 2.0f * val       ); // layer55 size 1*3x4 from layer54 convolve {1}
        ASSERT_FEQ(myNet.layers[56]->neurons[0][0].output, val =        val       ); // layer56 size 2*3x4 from layer55 pool max 1x1
        ASSERT_FEQ(myNet.layers[57]->neurons[0][0].output, val = 2.0f * val       ); // layer57 size 1*3x4 from layer56 convolve 1x1
        ASSERT_FEQ(myNet.layers[58]->neurons[0][0].output, val =        val       ); // layer58 from layer57 pool max 1x1
        ASSERT_FEQ(myNet.layers[59]->neurons[0][0].output, val = 1.0f * val       ); // layer59 from layer58 convolve 1x1
        ASSERT_FEQ(myNet.layers[60]->neurons[0][0].output, val =        val       ); // layer60 size 2*3x4 from layer59 pool max 1x1
        ASSERT_FEQ(myNet.layers[61]->neurons[0][0].output, val = 1.0f * val       ); // layer61 from layer60 convolve 1x1
        ASSERT_FEQ(myNet.layers[62]->neurons[0][0].output, val =        val       ); // layer62 size 1*3x4 from layer61 pool max 1x1
        ASSERT_FEQ(myNet.layers[63]->neurons[0][0].output, val = 1.0f * val       ); // layer63 size 2*3x4 from layer62 convolve 1x1
        ASSERT_FEQ(myNet.layers[64]->neurons[0][0].output, val =        val       ); // layer64 from layer63 pool max 1x1
        ASSERT_FEQ(myNet.layers[65]->neurons[0][0].output, val = 1.0f * val       ); // layer65 from layer64 convolve {1}
        ASSERT_FEQ(myNet.layers[66]->neurons[0][0].output, val =        val       ); // layer66 from layer65 pool max 1x1
        ASSERT_FEQ(myNet.layers[67]->neurons[0][0].output, val = 2.0f * val + 1.0f); // output size 3x4 from layer66 radius 0x0 tf linear    }
    }
}


void unitTestSparseConnections()
{
    {
        LOG("radius parameter");
        string config =
            "input size 10x10\n"
            "output size 8x8 from input radius 0x0\n";

        istringstream ss(config);
        Net myNet("", false);
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 10*10);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 8*8);

        const Neuron *pNeuron = &myNet.layers[0]->neurons[0][0];
        ASSERT_EQ(pNeuron->forwardConnectionsIndices.size(), 1);
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 0);

        ASSERT_EQ(myNet.connections.size(), 8*8 + 8*8);
    }

    {
        LOG("radius parameter 2");
        string config =
            "input size 10x10\n"
            "output size 1 from input radius 0x0\n";

        istringstream ss(config);
        Net myNet("", false);
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 10*10);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 1);

        const Neuron *pNeuron = &myNet.layers[0]->neurons[0][0];
        ASSERT_EQ(pNeuron->forwardConnectionsIndices.size(), 0);
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 0);

        ASSERT_EQ(myNet.connections.size(), 1 + 1);
    }

    {
        LOG("radius parameter 3");
        string config =
            "input size 10x10\n"
            "output size 1 from input radius 1x0\n";

        istringstream ss(config);
        Net myNet("", false);
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 10*10);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 1); // depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 1);
        ASSERT_EQ(myNet.connections.size(), 3 + 1);
    }

    {
        LOG("radius parameter 4");
        string config =
            "input size 10x10\n"
            "output size 1 from input radius 1x1\n";

        istringstream ss(config);
        Net myNet("", false);
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));
        ASSERT_EQ(myNet.connections.size(), 5 + 1); // Assumes elliptical projection
    }

    {
        LOG("radius parameter rectangular projection");
        string config =
            "input size 10x10\n"
            "output size 1 from input radius 1x1\n";

        istringstream ss(config);
        Net myNet("", false);
        myNet.projectRectangular = true;
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));
        ASSERT_EQ(myNet.connections.size(), 9 + 1); // Assumes elliptical projection
    }

    {
        LOG("kernel radius 1x0");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output size 8x8 from input radius 1x0 tf linear\n"; // 3 cols, one row

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);

        ASSERT_EQ(data[flattenXY(3, 1, 8)], pixelToNetworkInputRange(2));
        ASSERT_EQ(data[flattenXY(4, 1, 8)], pixelToNetworkInputRange(2));
        ASSERT_EQ(data[flattenXY(5, 1, 8)], pixelToNetworkInputRange(2));

        myNet.feedForward(myNet.sampleSet.samples[0]);
        auto const &inputLayer = *myNet.layers[0];
        auto const &outputLayer = *myNet.layers.back();

        ASSERT_EQ(inputLayer.neurons[0][flattenXY(3, 1, 8)].output, pixelToNetworkInputRange(2));
        ASSERT_EQ(inputLayer.neurons[0][flattenXY(4, 1, 8)].output, pixelToNetworkInputRange(2));
        ASSERT_EQ(inputLayer.neurons[0][flattenXY(5, 1, 8)].output, pixelToNetworkInputRange(2));

        // Output neurons at row 1 cover a row of all 2's, so its output should be
        // the sum of three pixels of value 2 plus a bias:

        ASSERT_EQ(outputLayer.neurons[0][flattenXY(3, 1, 8)].output, 3 * pixelToNetworkInputRange(2) + 1.0);
        ASSERT_EQ(outputLayer.neurons[0][flattenXY(6, 1, 8)].output, 3 * pixelToNetworkInputRange(2) + 1.0);
    }

    {
        LOG("radius 0x1");

        string topologyConfig =
            "input size 8x8 channel G\n"
            "output size 8x8 from input radius 0x1 tf linear\n"; // One col, 3 rows

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        // Neurons on column 1 cover a column of all 2's, so its output should be the sum
        // of three pixels of value 2 plus a bias:
        ASSERT_EQ(myNet.layers.back()->neurons[0][flattenXY(1, 3, 8)].output,
                    3 * pixelToNetworkInputRange(2) + 1.0);

        ASSERT_EQ(myNet.layers.back()->neurons[0][flattenXY(1, 5, 8)].output,
                    3 * pixelToNetworkInputRange(2) + 1.0);

        // Neuron on column 7 covers a col of all 8's, so its output should be the sum
        // of three pixels of value 8 plus a bias:
        ASSERT_EQ(myNet.layers.back()->neurons[0][flattenXY(7, 2, 8)].output,
                    3 * pixelToNetworkInputRange(8) + 1.0);
    }
}


void unitTestConvolutionFiltering()
{
    {
        LOG("Convolution filter {}");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "layer1 size 1x1 from input convolve {0.5} tf linear\n"
            "output size 1 from layer1\n";

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);

        auto const &h1 = *myNet.layers[1]; // the convolution filter layer
        ASSERT_EQ(h1.neurons.size(), 1); // depth = 1
        ASSERT_EQ(h1.neurons[0].size(), 1);
        ASSERT_EQ(h1.isConvolutionFilterLayer, true);
        ASSERT_EQ(h1.isPoolingLayer, false);
        ASSERT_EQ(h1.flatConvolveMatrix.size(), 1); // depth = 1
        ASSERT_EQ(h1.flatConvolveMatrix[0].size(), 1*1); // 1x1 kernel
        ASSERT_EQ(h1.neurons[0][0].forwardConnectionsIndices.size(), 1);

        auto const *pNeuron = &h1.neurons[0][0];
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 1); // no bias
        uint32_t backConnIdx = pNeuron->backConnectionsIndices[0];
        Connection const &conn = (*h1.pConnections)[backConnIdx];
        ASSERT_FEQ(h1.flatConvolveMatrix[0][conn.convolveMatrixIndex], 0.5f);
    }

    {
        LOG("Convolution filter 2x2");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "layer1 size 8x8 from input convolve {{0.25,0.5},{0,-0.25}} tf linear\n"
            "output size 1 from layer1\n";

        string inputDataConfig =
            "../images/8x8-test11.bmp 1\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        auto const &conv = *myNet.layers[1]; // the convolution filter layer
        ASSERT_EQ(conv.neurons.size(), 1); // depth = 1
        ASSERT_EQ(conv.neurons[0].size(), 8*8);
        ASSERT_EQ(conv.isConvolutionFilterLayer, true);
        ASSERT_EQ(conv.isPoolingLayer, false);
        ASSERT_EQ(conv.flatConvolveMatrix.size(), 1); // depth = 1
        ASSERT_EQ(conv.flatConvolveMatrix[0].size(), 2*2); // 2x2 kernel

        auto const &layer1 = *myNet.layers[1]; // convolution filter layer
        Neuron const &neuron1024 = layer1.neurons[0][flattenXY(2, 4, 8)];

        // check back connections to source neurons and their convolve matrix indices:
        ASSERT_EQ(neuron1024.backConnectionsIndices.size(), 4);

        Connection const &backConn00 =
                myNet.connections[neuron1024.backConnectionsIndices[flattenXY(0, 0, 2)]]; // top left
        ASSERT_EQ(backConn00.convolveMatrixIndex, (int)flattenXY(0,0,2));
        ASSERT_FEQ(layer1.flatConvolveMatrix[0][backConn00.convolveMatrixIndex], 0.25f);
        // neuron on layer 1 at 2,4 covers a 2x2 patch of neurons on layer 1 at 1,3, 2,3, 1,4, 2,4
        ASSERT_EQ(&backConn00.fromNeuron, &myNet.layers[0]->neurons[0][flattenXY(1,3,8)]);

        Connection const &backConn10 =
                myNet.connections[neuron1024.backConnectionsIndices[flattenXY(1, 0, 2)]]; // top right
        ASSERT_EQ(backConn10.convolveMatrixIndex, (int)flattenXY(1,0,2));
        ASSERT_FEQ(layer1.flatConvolveMatrix[0][backConn10.convolveMatrixIndex], 0.5f);
        ASSERT_EQ(&backConn10.fromNeuron, &myNet.layers[0]->neurons[0][flattenXY(2,3,8)]);

        Connection const &backConn01 =
                myNet.connections[neuron1024.backConnectionsIndices[flattenXY(0, 1, 2)]]; // bottom left
        ASSERT_EQ(backConn01.convolveMatrixIndex, (int)flattenXY(0,1,2));
        ASSERT_FEQ(layer1.flatConvolveMatrix[0][backConn01.convolveMatrixIndex] + 1.0f, 0.0f + 1.0f); // Avoid divide by zero
        ASSERT_EQ(&backConn01.fromNeuron, &myNet.layers[0]->neurons[0][flattenXY(1,4,8)]);

        Connection const &backConn11 =
                myNet.connections[neuron1024.backConnectionsIndices[flattenXY(1, 1, 2)]]; // bottom right
        ASSERT_EQ(backConn11.convolveMatrixIndex, (int)flattenXY(1,1,2));
        ASSERT_FEQ(layer1.flatConvolveMatrix[0][backConn11.convolveMatrixIndex], -0.25f);
        ASSERT_EQ(&backConn11.fromNeuron, &myNet.layers[0]->neurons[0][flattenXY(2,4,8)]);

        // layer 1 neuron at 2,4 covers four source neurons with outputs 4,4,5,5:
        float expected =
                  pixelToNetworkInputRange(4) * 0.25f
                + pixelToNetworkInputRange(4) * 0.5f
                + pixelToNetworkInputRange(5) * 0.0f
                + pixelToNetworkInputRange(5) * -0.25f;
        ASSERT_FEQ(neuron1024.output, expected);
    }
}


void unitTestConvolutionNetworking()
{
    {
        LOG("Convolution network trivial kernel 1x1 in 1x1 plane");
        /*
               /-1x1[1x1]-\
            1x1            1
               \-1x1[1x1]-/
         */

        string topologyConfig =
            "input size 1x1\n"
            "layerConv size 2*1x1 from input convolve 1x1 tf linear\n"
            "output size 1 from layerConv\n";

        string inputDataConfig =
            "{ 0.25 } 1.0\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);

        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // input layer depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 1*1);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 2); // hidden layer depth = 2
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 1*1);
        ASSERT_EQ(myNet.layers[1]->neurons[1].size(), 1*1);
        ASSERT_EQ(myNet.layers[2]->neurons.size(), 1); // output layer depth = 1
        ASSERT_EQ(myNet.layers[2]->neurons[0].size(), 1);

        auto const &hl = *myNet.layers[1]; // Hidden layer (the convolution network layer)
        ASSERT_EQ(hl.size.depth, 2);
        ASSERT_EQ(hl.size.x, 1);
        ASSERT_EQ(hl.size.y, 1);

        ASSERT_EQ(hl.flatConvolveMatrix.size(), 2); // depth = 2
        ASSERT_EQ(hl.flatConvolveMatrix[0].size(), 1*1); // kernel size 1x1
        ASSERT_EQ(hl.flatConvolveMatrix[1].size(), 1*1);

        ASSERT_EQ(hl.isConvolutionFilterLayer, false);
        ASSERT_EQ(hl.isConvolutionNetworkLayer, true);
        ASSERT_EQ(hl.isPoolingLayer, false);
        ASSERT_EQ(hl.kernelSize.x, 1);
        ASSERT_EQ(hl.kernelSize.y, 1);

        auto &n000 = hl.neurons[0][flattenXY(0, 0, hl.size)];   // depth,x,y = 0,0,0
        auto &n100 = hl.neurons[1][flattenXY(0, 0, hl.size)];   // depth,x,y = 1,0,0

        ASSERT_EQ(n000.backConnectionsIndices.size(), 1); // 1 source, no bias
        ASSERT_EQ(n100.backConnectionsIndices.size(), 1); // 1 source, no bias

        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        // The sole hidden-layer neuron covers the sole input neuron, which has value 0.25:
        ASSERT_EQ(myNet.layers[0]->neurons[0][flattenXY(0, 0, myNet.layers[0]->size)].output, 0.25);
        auto backIndex = n000.backConnectionsIndices[0];    // source neuron index
        auto &sourceNeuron = myNet.connections[backIndex].fromNeuron;
        ASSERT_EQ(sourceNeuron.output, 0.25);
        ASSERT_EQ(hl.flatConvolveMatrix.size(), 2);    // depth = 2
        ASSERT_EQ(hl.flatConvolveMatrix[0].size(), 1*1); // 1*1 kernel elements, depth 0
        ASSERT_EQ(hl.flatConvolveMatrix[1].size(), 1*1); // 1*1 kernel elements, depth 1
        ASSERT_EQ(n000.output, 0.25);  // depth,x,y = 0,0,0, no bias
        ASSERT_EQ(n100.output, 0.25);  // depth,x,y = 1,0,0, no bias
    }

    {
        LOG("Convolution network kernel 1x1 in 8x8 plane");
        /*
               /-8x8[1x1]-\
            8x8            1
               \-8x8[1x1]-/
         */

        string topologyConfig =
            "input size 8x8 channel R\n"
            "layerConv size 2*8x8 from input convolve 1x1 tf linear\n"
            "output size 1 from layerConv\n";

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);

        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1); // input layer depth = 1
        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 8*8);
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 2); // hidden layer depth = 2
        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 8*8);
        ASSERT_EQ(myNet.layers[1]->neurons[1].size(), 8*8);
        ASSERT_EQ(myNet.layers[2]->neurons.size(), 1); // output layer depth = 1
        ASSERT_EQ(myNet.layers[2]->neurons[0].size(), 1);

        auto const &hl = *myNet.layers[1]; // Hidden layer (the convolution network layer)
        ASSERT_EQ(hl.size.depth, 2);
        ASSERT_EQ(hl.size.x, 8);
        ASSERT_EQ(hl.size.y, 8);

        ASSERT_EQ(hl.flatConvolveMatrix.size(), 2); // depth = 2
        ASSERT_EQ(hl.flatConvolveMatrix[0].size(), 1*1); // kernel size 1x1
        ASSERT_EQ(hl.flatConvolveMatrix[1].size(), 1*1);

        ASSERT_EQ(hl.isConvolutionFilterLayer, false);
        ASSERT_EQ(hl.isConvolutionNetworkLayer, true);
        ASSERT_EQ(hl.isPoolingLayer, false);

        ASSERT_EQ(hl.kernelSize.x, 1);
        ASSERT_EQ(hl.kernelSize.y, 1);

        auto &n000 = hl.neurons[0][flattenXY(0, 0, hl.size)];   // depth,x,y = 0,0,0
        auto &n100 = hl.neurons[1][flattenXY(0, 0, hl.size)];   // depth,x,y = 1,0,0
        auto &n024 = hl.neurons[0][flattenXY(2, 4, hl.size)];   // depth,x,y = 0,2,4

        ASSERT_EQ(n000.backConnectionsIndices.size(), 1); // 1 source, no bias
        ASSERT_EQ(n100.backConnectionsIndices.size(), 1); // 1 source, no bias
        ASSERT_EQ(n024.backConnectionsIndices.size(), 1); // 1 source, no bias

        setAllWeights(myNet, 1.0);
        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        ASSERT_EQ(myNet.layers[0]->neurons[0][flattenXY(2, 4,8)].output, pixelToNetworkInputRange(5));

        // Each hidden-layer neuron covers only one input neuron, which has value based on its row number:
        ASSERT_EQ(myNet.layers[0]->neurons[0][flattenXY(0, 0, myNet.layers[0]->size)].output,
                    pixelToNetworkInputRange(1));
        ASSERT_EQ(myNet.layers[0]->neurons[0][flattenXY(2, 4, myNet.layers[0]->size)].output,
                    pixelToNetworkInputRange(5));

        auto backIndex = n000.backConnectionsIndices[0];    // source neuron index
        auto &sourceNeuron = myNet.connections[backIndex].fromNeuron;

        ASSERT_EQ(sourceNeuron.output, pixelToNetworkInputRange(1));

        ASSERT_EQ(hl.flatConvolveMatrix.size(), 2);    // depth = 2
        ASSERT_EQ(hl.flatConvolveMatrix[0].size(), 1*1); // 1*1 kernel elements, depth 0
        ASSERT_EQ(hl.flatConvolveMatrix[1].size(), 1*1); // 1*1 kernel elements, depth 1

        ASSERT_EQ(n000.output, pixelToNetworkInputRange(1));  // depth,x,y = 0,0,0, no bias
        ASSERT_EQ(n100.output, pixelToNetworkInputRange(1));  // depth,x,y = 1,0,0, no bias
        ASSERT_EQ(n024.output, pixelToNetworkInputRange(5));  // depth,x,y = 0,2,4, no bias
    }

    {
        LOG("Convolution networking");

        /*
                    /8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--\
                   / 8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]-- \
                  /  8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--  \
                 /   8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--   \
            1*8x8    8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--    1*10x1--1*10x1--1*3x1
                 \   8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--   /
                  \  8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--  /
                   \ 8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]-- /
                    \8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--/
                     8x8[2x2]b--6x6[2x2]--6x6[2x2]b--4x4[2x2]--
         */

        string topologyConfig =
            "input size 8x8\n"
            "layerConvolve1 size 10*8x8 from input convolve 2x2\n"
            "layerPool1 size 10*6x6 from layerConvolve1 pool max 2x2\n"
            "layerConvolve2 size 10*6x6 from layerPool1 convolve 2x2\n"
            "layerPool2 size 10*4x4 from layerConvolve2 pool max 2x2\n"
            "layerReduce size 10 from layerPool2\n"
            "layerHidden size 10 from layerReduce\n"
            "output size 3 from layerHidden\n";

        string inputDataConfig =
            "../images/8x8-test.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        // verify some structural stuff:
        ASSERT_EQ(myNet.layers.size(), 8);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1);  // input size 1*8x8
        ASSERT_EQ(layerNamed(myNet, "layerConvolve1")->neurons.size(), 10); // layerConvolve1 size 10*8x8
        ASSERT_EQ(layerNamed(myNet, "layerPool1")->neurons.size(), 10);     // layerPool1 size 10*6x6
        ASSERT_EQ(layerNamed(myNet, "layerConvolve2")->neurons.size(), 10); // layerConvolve2 size 10*6x6
        ASSERT_EQ(layerNamed(myNet, "layerPool2")->neurons.size(), 10);     // layerPool2 size 10*4x4
        ASSERT_EQ(layerNamed(myNet, "layerReduce")->neurons.size(), 1);     // layerReduce size 10
        ASSERT_EQ(layerNamed(myNet, "layerHidden")->neurons.size(), 1);     // layerHidden size 10
        ASSERT_EQ(myNet.layers.back()->neurons.size(), 1);       // output size 3

        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 8*8); // input size 1*8x8
        ASSERT_EQ(layerNamed(myNet, "layerConvolve1")->neurons[0].size(), 8*8); // layerConvolve1 size 10*8x8
        ASSERT_EQ(layerNamed(myNet, "layerPool1")->neurons[0].size(), 6*6);     // layerPool1 size 10*6x6
        ASSERT_EQ(layerNamed(myNet, "layerConvolve2")->neurons[0].size(), 6*6); // layerConvolve2 size 10*6x6
        ASSERT_EQ(layerNamed(myNet, "layerPool2")->neurons[0].size(), 4*4);     // layerPool2 size 10*4x4
        ASSERT_EQ(layerNamed(myNet, "layerReduce")->neurons[0].size(), 10);     // layerReduce size 10
        ASSERT_EQ(layerNamed(myNet, "layerHidden")->neurons[0].size(), 10);     // layerHidden size 10
        ASSERT_EQ(myNet.layers.back()->neurons[0].size(), 3);                   // output size 3

        // To do:  more tests!
    }

    {
        LOG("Convolution networking backprop");

        /*
               /-8x8[1x1]-\
            8x8            10
               \-8x8[1x1]-/
         */

        string topologyConfig =
            "input size 8x8 channel B\n"
            "layerConv size 2*8x8 from input convolve 1x1 tf linear\n"
            "output size 1 from layerConv tf linear\n";

        string inputDataConfig =
            "../images/8x8-test.bmp 1.0\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        // verify some structural stuff:
        ASSERT_EQ(myNet.layers.size(), 3);
        ASSERT_EQ(myNet.layers[0]->neurons.size(), 1);  // input depth = 1
        ASSERT_EQ(myNet.layers[1]->neurons.size(), 2);  // layerConv depth = 2
        ASSERT_EQ(myNet.layers[2]->neurons.size(), 1);  // output depth = 1

        ASSERT_EQ(myNet.layers[0]->neurons[0].size(), 8*8); // input size 1*8x8

        ASSERT_EQ(myNet.layers[1]->neurons[0].size(), 8*8); // layerConv size 2*8x8
        ASSERT_EQ(myNet.layers[1]->neurons[1].size(), 8*8);

        ASSERT_EQ(myNet.layers[2]->neurons[0].size(), 1);   // output size 1

        myNet.backProp(myNet.sampleSet.samples[0]);
    }

    {
        LOG("Convolution networking split topology");

        /*
                   /-20*32x32[7x7]--20*8x8[2x2]--\
            1*32x32                               1*8x8--1*4x4--1*10x1
                   \                             /
                    \-1*8x8[1x3]----------------/
         */

        string topologyConfig =
            "input size 32x32\n"
            "layerConv size 10*32x32 from input convolve 7x7\n"
            "layerPool size 10*8x8 from layerConv pool max 2x2\n"
            "layerMix1 size 8x8 from layerPool\n"
            "layerGauss size 8x8 from input radius 1x3 tf gaussian\n"
            "layerMix2 size 4x4 from layerMix1\n"
            "layerMix2 size 4x4 from layerGauss\n"
            "output size 10 from layerMix2\n";

        string inputDataConfig =
            "../images/8x8-test.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        const string filename = "./unitTestSavedWeights.txt";

        Net myNet1(topologyConfigFilename, false);
        myNet1.sampleSet.loadSamples(inputDataConfigFilename);

        // move the following topological tests to a different unit test
        ASSERT_EQ(myNet1.layers[0]->neurons[0].size(), 32*32);

        auto const &input = *myNet1.layers[0];
        ASSERT_EQ(input.size.depth, 1);
        ASSERT_EQ(input.neurons.size(), 1);
        ASSERT_EQ(input.neurons[0].size(), 32*32);
        ASSERT_EQ(input.neurons[0][0].backConnectionsIndices.size(), 0);
        ASSERT_EQ(input.neurons[0].back().backConnectionsIndices.size(), 0);
        ASSERT_EQ(input.neurons[0][7*16+7].forwardConnectionsIndices.size(), 10*7*7);

        // layerMix2 combines two source layers:
        auto const &layerMix2 = *layerNamed(myNet1, "layerMix2"); // size 4x4 from layerMix1 plus from layerGauss
        ASSERT_EQ(layerMix2.size.depth, 1);
        ASSERT_EQ(layerMix2.neurons.size(), 1);
        ASSERT_EQ(layerMix2.neurons[0].size(), 4*4);
        ASSERT_EQ(layerMix2.neurons[0][flattenXY(2,2,4)].backConnectionsIndices.size(),
                  8*8 + 8*8 + 1); // two source layers plus a bias

        auto const &layerConv = *layerNamed(myNet1, "layerConv"); // size 10*32x32 from input convolve 7x7
        ASSERT_EQ(layerConv.size.depth, 10);
        ASSERT_EQ(layerConv.neurons.size(), 10);
        ASSERT_EQ(layerConv.neurons[0].size(), 32*32);
        ASSERT_EQ(layerConv.neurons[0][16*32+16].backConnectionsIndices.size(), 7*7);

        auto const &output = *myNet1.layers.back();
        ASSERT_EQ(output.size.depth, 1);
        ASSERT_EQ(output.neurons.size(), 1);
        ASSERT_EQ(output.neurons[0].size(), 10);
        ASSERT_EQ(output.neurons[0][0].backConnectionsIndices.size(), 4*4 + 1);
        ASSERT_EQ(output.neurons[0].back().backConnectionsIndices.size(), 4*4 + 1);
        ASSERT_EQ(output.neurons[0][0].forwardConnectionsIndices.size(), 0);
        ASSERT_EQ(output.neurons[0].back().forwardConnectionsIndices.size(), 0);
    }
}


void unitTestImages()
{
    {
        LOG("images smoke test");

        // to do: We should verify here that the topology.txt and inputData.txt
        // files are the default version, and that the digits archive has been
        // expanded in the usual subdirectory

        Net myNet("../images/digits/topology.txt", false);
        ASSERT_EQ(myNet.layers.size(), 3);
        ASSERT_EQ(myNet.layers[0]->size.x, 32);
        ASSERT_EQ(myNet.layers[0]->size.y, 32);
        ASSERT_EQ(myNet.sampleSet.samples.size(), 0);

        myNet.sampleSet.loadSamples("../images/digits/inputData.txt");
        ASSERT_EQ(myNet.sampleSet.samples.size(), 5000);
    }

    {
        LOG("8x8-test.bmp orientation");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output size 1 from input radius 0x0\n";

        string inputDataConfig =
            "../images/8x8-test.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        ASSERT_EQ(myNet.connections.size(), 1+1); // incl bias

        setAllWeights(myNet, 1.0);
        ASSERT_EQ((int)(myNet.connections[0].weight * 100 + 0.5), 100);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        ASSERT_EQ(myNet.sampleSet.samples.size(), 1);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        ASSERT_EQ(data.size(), 8*8);
        ASSERT_EQ(data[flattenXY(0, 0, 8)], pixelToNetworkInputRange(10));
        ASSERT_EQ(data[flattenXY(0, 1, 8)], pixelToNetworkInputRange(101));
        ASSERT_EQ(data[flattenXY(0, 7, 8)], pixelToNetworkInputRange(30));
        ASSERT_EQ(data[flattenXY(1, 0, 8)], pixelToNetworkInputRange(101));
        ASSERT_EQ(data[flattenXY(7, 0, 8)], pixelToNetworkInputRange(20));
        ASSERT_EQ(data[flattenXY(7, 7, 8)], pixelToNetworkInputRange(40));
    }

    {
        LOG("8x8-test11.bmp channels");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output size 1 from input tf linear\n"; // With weights=1, this will sum the inputs+bias

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        ASSERT_EQ(myNet.connections.size(), 8*8 + 1);

        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        ASSERT_EQ(data[flattenXY(0, 0, 8)], pixelToNetworkInputRange(1));
        ASSERT_EQ(data[flattenXY(0, 1, 8)], pixelToNetworkInputRange(2));
        ASSERT_EQ(data[flattenXY(1, 0, 8)], pixelToNetworkInputRange(1));
        ASSERT_EQ(data[flattenXY(1, 7, 8)], pixelToNetworkInputRange(8));
        ASSERT_EQ(data[flattenXY(2, 4, 8)], pixelToNetworkInputRange(5));

        // Output should be the sum of 64 input values of pixels with values from 1..8.
        //   plus a bias input of 1.0:
        float avgInputVal = (pixelToNetworkInputRange(1) + pixelToNetworkInputRange(8)) / 2.0f;
        ASSERT_EQ(myNet.layers.back()->neurons[0][0].output, 64 * avgInputVal + 1.0);

        myNet.sampleSet.clearImageCache();
        data = myNet.sampleSet.samples[0].getData(NNet::R);
        ASSERT_EQ(data[flattenXY(0, 0, 8)], pixelToNetworkInputRange(1));
        ASSERT_EQ(data[flattenXY(0, 1, 8)], pixelToNetworkInputRange(2));
        ASSERT_EQ(data[flattenXY(1, 0, 8)], pixelToNetworkInputRange(1));
        ASSERT_EQ(data[flattenXY(1, 7, 8)], pixelToNetworkInputRange(8));
        ASSERT_EQ(data[flattenXY(2, 4, 8)], pixelToNetworkInputRange(5));

        myNet.sampleSet.clearImageCache();
        data = myNet.sampleSet.samples[0].getData(NNet::G);
        ASSERT_EQ(data[flattenXY(0, 0, 8)], pixelToNetworkInputRange(1));
        ASSERT_EQ(data[flattenXY(0, 1, 8)], pixelToNetworkInputRange(1));
        ASSERT_EQ(data[flattenXY(1, 0, 8)], pixelToNetworkInputRange(2));
        ASSERT_EQ(data[flattenXY(1, 7, 8)], pixelToNetworkInputRange(2));
        ASSERT_EQ(data[flattenXY(7, 7, 8)], pixelToNetworkInputRange(8));
        ASSERT_EQ(data[flattenXY(2, 4, 8)], pixelToNetworkInputRange(3));

        myNet.sampleSet.clearImageCache();
        data = myNet.sampleSet.samples[0].getData(NNet::B);
        ASSERT_EQ(data[flattenXY(0, 0, 8)], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[flattenXY(0, 1, 8)], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[flattenXY(1, 0, 8)], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[flattenXY(1, 7, 8)], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[flattenXY(2, 4, 8)], pixelToNetworkInputRange(127));
    }

    {
        LOG("Image read and orientation");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output size 1 from input radius 0x0 tf linear\n";

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        ASSERT_EQ(data[flattenXY(3, 1, 8)], pixelToNetworkInputRange(2));
        ASSERT_EQ(data[flattenXY(2, 4, 8)], pixelToNetworkInputRange(5));

        auto const &inputLayer = *myNet.layers[0];
        ASSERT_EQ(inputLayer.neurons[0][flattenXY(3,1,8)].output, pixelToNetworkInputRange(2));
        ASSERT_EQ(inputLayer.neurons[0][flattenXY(2,4,8)].output, pixelToNetworkInputRange(5));

        // Output neurons at row 1 covers a single neuron, at x=3 or 4 and y = 3 or 4,
        // depending on roundoff (one of the following lines will be correct):

        auto const &outputLayer = *myNet.layers.back();
        ASSERT_EQ(outputLayer.neurons[0][0].output, 1*1 * pixelToNetworkInputRange(5) + 1.0);
    }

    {
        LOG("input data config file, path_prefix directive");

        string topologyConfig =
            "input size 1\n"
            "output from input\n";

        string inputDataConfig =
            "path_prefix = ../images/\n"
            "8x8-test.bmp\n"
            "path_prefix=\n"
            "../images/8x8-test.bmp\n"
            "path_prefix =../images/\n"
            "8x8-test.bmp\n"
            "8x8-test.bmp\n"
            "path_prefix=\n"
            "../images/8x8-test.bmp\n"
            "path_prefix =\n"
            "../images/8x8-test.bmp\n"
            "path_prefix = \n"
            "../images/8x8-test.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        myNet.sampleSet.loadSamples(inputDataConfigFilename);

        for (auto const &sample : myNet.sampleSet.samples) {
            ASSERT_EQ(sample.imageFilename, "../images/8x8-test.bmp");
        }
    }

    {
        LOG("8x8-test.dat orientation channel R single precision");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output from input radius 0x0\n";

        string inputDataConfig =
            "../images/8x8-test.dat\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        auto const &inputs = myNet.layers[0]->neurons[0];
        ASSERT_FEQ(inputs[flattenXY(0,0,8)].output, pixelToNetworkInputRange(10));
        ASSERT_FEQ(inputs[flattenXY(7,0,8)].output, pixelToNetworkInputRange(20));
        ASSERT_FEQ(inputs[flattenXY(0,7,8)].output, pixelToNetworkInputRange(30));
        ASSERT_FEQ(inputs[flattenXY(7,7,8)].output, pixelToNetworkInputRange(40));
        ASSERT_FEQ(inputs[flattenXY(2,4,8)].output, pixelToNetworkInputRange(101));
    }

    {
        LOG("8x8-test.dat orientation channel G double precision");

        string topologyConfig =
            "input size 8x8 channel G\n"
            "output from input radius 0x0\n";

        string inputDataConfig =
            "../images/8x8-test-doubleprecision.dat\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        auto const &inputs = myNet.layers[0]->neurons[0];
        ASSERT_FEQ(inputs[flattenXY(0,0,8)].output, pixelToNetworkInputRange(11));
        ASSERT_FEQ(inputs[flattenXY(7,0,8)].output, pixelToNetworkInputRange(21));
        ASSERT_FEQ(inputs[flattenXY(0,7,8)].output, pixelToNetworkInputRange(31));
        ASSERT_FEQ(inputs[flattenXY(7,7,8)].output, pixelToNetworkInputRange(41));
        ASSERT_FEQ(inputs[flattenXY(2,4,8)].output, pixelToNetworkInputRange(101));
    }
}


void unitTestPooling()
{
    {
        LOG("Pooling trivial config");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "layerPool size 1 from input pool max 1x1 tf linear\n"
            "output from layerPool tf linear\n";

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        ASSERT_EQ(myNet.layers.size(), 3);
        auto const &pl = *myNet.layers[1]; // the pooling layer
        ASSERT_EQ(pl.neurons.size(), 1); // depth = 1
        ASSERT_EQ(pl.neurons[0].size(), 1*1);

        ASSERT_EQ(pl.flatConvolveMatrix.size(), 0);
        ASSERT_EQ(pl.isConvolutionFilterLayer, false);
        ASSERT_EQ(pl.isConvolutionNetworkLayer, false);
        ASSERT_EQ(pl.isPoolingLayer, true);
        ASSERT_EQ(pl.poolMethod, NNet::POOL_MAX);
        ASSERT_EQ(pl.poolSize.x, 1);
        ASSERT_EQ(pl.poolSize.y, 1);
        ASSERT_EQ(pl.size.depth, 1);
        ASSERT_EQ(pl.size.x, 1);
        ASSERT_EQ(pl.size.y, 1);

        // The pooling layer neuron at 0,0 covers an input neuron on either row 3 or
        // 4 depending on roundoff, with pixel value 4 or 5:
        // No bias, and that's the only input so it's the max:
        auto const &n00 = pl.neurons[0][0];
        ASSERT_EQ(n00.backConnectionsIndices.size(), 1);
        auto backIdx = n00.backConnectionsIndices[0];
        auto const &backConn = myNet.connections[backIdx];
        auto const &sourceNeuron = backConn.fromNeuron;
        ASSERT_EQ(sourceNeuron.output, pixelToNetworkInputRange(5));
        ASSERT_EQ(n00.output, pixelToNetworkInputRange(5));

        ASSERT_EQ(n00.forwardConnectionsIndices.size(), 1);
        auto const &outputNeuron = myNet.layers.back()->neurons[0][0];
        ASSERT_EQ(outputNeuron.output, n00.output + 1.0);
    }

    {
        LOG("Pool max 1*8x8 to 1*2x2[4x4]");

        // Channel B contains pixels all of value 127. In this test, we use
        // pooling to find the max or avg of the four quadrants of the input.

        string topologyConfig =
            "input size 8x8 channel B\n"
            "layerPool size 2x2 from input pool max 4x4 tf linear\n"
            "output size 1 from layerPool tf linear\n";

        string inputDataConfig =
            "../images/8x8-test11.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::B);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        ASSERT_EQ(myNet.layers.size(), 3);
        auto const &pl = *myNet.layers[1]; // the pooling layer
        ASSERT_EQ(pl.neurons.size(), 1); // depth = 1
        ASSERT_EQ(pl.neurons[0].size(), 2*2);

        // The NW (upper left) pooling layer neuron covers the upper left quadrant of the input image
        auto const &neuronNW = pl.neurons[0][flattenXY(0,0,2)];
        auto const &neuronNE = pl.neurons[0][flattenXY(1,0,2)];
        auto const &neuronSW = pl.neurons[0][flattenXY(0,1,2)];
        auto const &neuronSE = pl.neurons[0][flattenXY(1,1,2)];

        // each pooling neuron covers a 4x4 patch of input neurons:
        ASSERT_EQ(neuronNW.backConnectionsIndices.size(), 4*4);
        ASSERT_EQ(neuronNE.backConnectionsIndices.size(), 4*4);
        ASSERT_EQ(neuronSW.backConnectionsIndices.size(), 4*4);
        ASSERT_EQ(neuronSE.backConnectionsIndices.size(), 4*4);

        // let's verify some raw inputs of the NW quadrant:
        ASSERT_EQ(data[flattenXY(0,0,8)], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[flattenXY(3,0,8)], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[flattenXY(3,3,8)], pixelToNetworkInputRange(127));

        // NW quadrant max:
        float expectedOutput = pixelToNetworkInputRange(127); // max
        ASSERT_EQ(neuronNW.output, expectedOutput);

        // NE quadrant max:
        ASSERT_EQ(neuronNE.output, expectedOutput);

        // SW quadrant max:
        ASSERT_EQ(neuronSW.output, expectedOutput);

        // SE quadrant max:
        ASSERT_EQ(neuronSE.output, expectedOutput);
    }

    {
        LOG("Pool avg 1*8x8 to 1*2x2[4x4]");

        // In this test, we use pooling to find the average in each of the
        // four quadrants of the input.

        string topologyConfig =
            "input size 8x8 channel R\n"
            "layerPool size 2x2 from input pool avg 4x4 tf linear\n"
            "output size 1 from layerPool tf linear\n";

        string inputDataConfig =
            "../images/8x8-test.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        auto const &pl = *myNet.layers[1]; // the pooling layer

        // The NW (upper left) pooling layer neuron covers the upper left quadrant of the input image
        auto const &neuronNW = pl.neurons[0][flattenXY(0,0,2)];
        auto const &neuronNE = pl.neurons[0][flattenXY(1,0,2)];
        auto const &neuronSW = pl.neurons[0][flattenXY(0,1,2)];
        auto const &neuronSE = pl.neurons[0][flattenXY(1,1,2)];

        // let's verify some raw inputs of the NW quadrant:
        ASSERT_EQ(data[flattenXY(0,0,8)], pixelToNetworkInputRange(10));
        ASSERT_EQ(data[flattenXY(3,0,8)], pixelToNetworkInputRange(101));
        ASSERT_EQ(data[flattenXY(3,3,8)], pixelToNetworkInputRange(101));

        // NW quadrant avg:
        float expectedOutput = (pixelToNetworkInputRange(10) + 15 * pixelToNetworkInputRange(101)) / 16.0f;
        ASSERT_EQ(neuronNW.output, expectedOutput);

        // NE quadrant avg:
        expectedOutput = (pixelToNetworkInputRange(20) + 15 * pixelToNetworkInputRange(101)) / 16.0f;
        ASSERT_EQ(neuronNE.output, expectedOutput);

        // SW quadrant avg:
        expectedOutput = (pixelToNetworkInputRange(30) + 15 * pixelToNetworkInputRange(101)) / 16.0f;
        ASSERT_EQ(neuronSW.output, expectedOutput);

        // SE quadrant avg:
        expectedOutput = (pixelToNetworkInputRange(40) + 15 * pixelToNetworkInputRange(101)) / 16.0f;
        ASSERT_EQ(neuronSE.output, expectedOutput);
    }

    {
        LOG("Pool avg 2*8x8 to 2*2x2[4x4]");

        /*
                 /-8x8[1x1]b--2x2[4x4]-\
            1*8x8                       1*1x1
                 \-8x8[1x1]b--2x2[4x4]-/
         */

        // In this test, we use pooling to find the average in each of the
        // four quadrants of the input, in a convolution network layer with depth = 2.

        string topologyConfig =
            "input size 8x8 channel R\n"
            "layerConvPassthrough size 2*8x8 from input convolve 1x1 tf linear\n"
            "layerPool size 2*2x2 from layerConvPassthrough pool avg 4x4 tf linear\n"
            "output size 1 from layerPool tf linear\n";

        string inputDataConfig =
            "../images/8x8-test.bmp\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename, false);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        auto const &pInput = myNet.layers.front();
        auto const *pConv = layerNamed(myNet, "layerConvPassthrough");
        auto const *pPool = layerNamed(myNet, "layerPool");
        auto const &pOutput = myNet.layers.back();

        // verify some structural stuff:
        ASSERT_EQ(myNet.layers.size(), 4);
        ASSERT_EQ(pInput->neurons.size(), 1); // depth = 1
        ASSERT_EQ(pConv->neurons.size(), 2); // depth = 2
        ASSERT_EQ(pPool->neurons.size(), 2); // depth = 2
        ASSERT_EQ(pOutput->neurons.size(), 1); // depth = 1

        ASSERT_EQ(pInput->neurons[0].size(), 8*8);
        ASSERT_EQ(pConv->neurons[0].size(), 8*8);
        ASSERT_EQ(pConv->neurons[1].size(), 8*8);
        ASSERT_EQ(pPool->neurons[0].size(), 2*2);
        ASSERT_EQ(pPool->neurons[1].size(), 2*2);
        ASSERT_EQ(pOutput->neurons[0].size(), 1);

        ASSERT_EQ(pInput->neurons[0][0].forwardConnectionsIndices.size(), 2); // input splits into two

        for (uint32_t nIdx = 0; nIdx < pConv->size.x * pConv->size.y; ++nIdx) {
            ASSERT_EQ(pConv->neurons[0][nIdx].backConnectionsIndices.size(), 1); // one source, no bias
            ASSERT_EQ(pConv->neurons[1][nIdx].backConnectionsIndices.size(), 1);
            ASSERT_EQ(pConv->neurons[0][nIdx].forwardConnectionsIndices.size(), 1);
            ASSERT_EQ(pConv->neurons[1][nIdx].forwardConnectionsIndices.size(), 1);
        }

        // layerPool size 2*2x2 from layerConvPassthrough pool avg 4x4
        for (uint32_t nIdx = 0; nIdx < pPool->size.x * pPool->size.y; ++nIdx) {
            ASSERT_EQ(pPool->neurons[0][nIdx].backConnectionsIndices.size(), 4*4); // no bias on pooling layers
            ASSERT_EQ(pPool->neurons[1][nIdx].backConnectionsIndices.size(), 4*4);
            ASSERT_EQ(pPool->neurons[0][nIdx].forwardConnectionsIndices.size(), 1);
            ASSERT_EQ(pPool->neurons[1][nIdx].forwardConnectionsIndices.size(), 1);
        }

        ASSERT_EQ(myNet.layers[3]->neurons[0][0].backConnectionsIndices.size(), 2*2*2 + 1);

        auto idx = pPool->neurons[0][flattenXY(0,0,2)].backConnectionsIndices[0];
        ASSERT_EQ(&myNet.connections[idx].fromNeuron, &pConv->neurons[0][flattenXY(0,0,8)]);
        idx = pPool->neurons[1][flattenXY(0,0,2)].backConnectionsIndices[0];
        ASSERT_EQ(&myNet.connections[idx].fromNeuron, &pConv->neurons[1][flattenXY(0,0,8)]);

        // The NW (upper left) pooling layer neuron covers the upper left quadrant of the input image
        auto const &neuronNW0 = pPool->neurons[0][flattenXY(0,0,2)];
        auto const &neuronNE0 = pPool->neurons[0][flattenXY(1,0,2)];
        auto const &neuronSW0 = pPool->neurons[0][flattenXY(0,1,2)];
        auto const &neuronSE0 = pPool->neurons[0][flattenXY(1,1,2)];

        auto const &neuronNW1 = pPool->neurons[1][flattenXY(0,0,2)];
        auto const &neuronNE1 = pPool->neurons[1][flattenXY(1,0,2)];
        auto const &neuronSW1 = pPool->neurons[1][flattenXY(0,1,2)];
        auto const &neuronSE1 = pPool->neurons[1][flattenXY(1,1,2)];

        // might as well verify some of the outputs of layerConvPassthrough
        ASSERT_EQ(pConv->neurons[0][flattenXY(0,0,8)].output, pixelToNetworkInputRange(10));
        ASSERT_EQ(pConv->neurons[1][flattenXY(0,0,8)].output, pixelToNetworkInputRange(10));
        ASSERT_EQ(pConv->neurons[0][flattenXY(1,1,8)].output, pixelToNetworkInputRange(101));
        ASSERT_EQ(pConv->neurons[1][flattenXY(1,1,8)].output, pixelToNetworkInputRange(101));

        // NW quadrant avg:
        float expectedOutput =
                ((pixelToNetworkInputRange(10) /*+ 1.0*/) + 15 * (pixelToNetworkInputRange(101) /*+ 1.0*/) ) / 16.0f;
        ASSERT_EQ(neuronNW0.output, expectedOutput);
        ASSERT_EQ(neuronNW1.output, expectedOutput);

        // NE quadrant avg:
        expectedOutput = ((pixelToNetworkInputRange(20) /*+ 1.0*/) + 15 * (pixelToNetworkInputRange(101) /*+ 1.0*/) ) / 16.0f;
        ASSERT_EQ(neuronNE0.output, expectedOutput);
        ASSERT_EQ(neuronNE1.output, expectedOutput);

        // SW quadrant avg:
        expectedOutput = ((pixelToNetworkInputRange(30) /*+ 1.0*/) + 15 * (pixelToNetworkInputRange(101) /*+ 1.0*/) ) / 16.0f;
        ASSERT_EQ(neuronSW0.output, expectedOutput);
        ASSERT_EQ(neuronSW1.output, expectedOutput);

        // SE quadrant avg:
        expectedOutput = ((pixelToNetworkInputRange(40) /*+ 1.0*/) + 15 * (pixelToNetworkInputRange(101) /*+ 1.0*/) ) / 16.0f;
        ASSERT_EQ(neuronSE0.output, expectedOutput);
        ASSERT_EQ(neuronSE1.output, expectedOutput);
    }
}


void unitTestMisc()
{
    // To do: add test for save/load weights

    {
        LOG("test sanitizeFilename");   // != '_' && c != '-' && c != '.' && c != '%')

        extern void sanitizeFilename(string &s);
        string s;
        s = ""; sanitizeFilename(s); ASSERT_EQ(s, "");
        s = "_"; sanitizeFilename(s); ASSERT_EQ(s, "_");
        s = "-"; sanitizeFilename(s); ASSERT_EQ(s, "-");
        s = "%"; sanitizeFilename(s); ASSERT_EQ(s, "%");
        s = "$"; sanitizeFilename(s); ASSERT_EQ(s, "_");
        s = "%%%%"; sanitizeFilename(s); ASSERT_EQ(s, "%%%%");
        s = "a-*bc&.%d"; sanitizeFilename(s); ASSERT_EQ(s, "a-_bc_.%d");
        s = "&^%$*)*_+"; sanitizeFilename(s); ASSERT_EQ(s, "__%______");
        s = " <"; sanitizeFilename(s); ASSERT_EQ(s, "__");
    }

    {
        LOG("index flattening");

        ASSERT_EQ(flattenXY(0, 0, 8), 0);
        ASSERT_EQ(flattenXY(0, 1, 8), 1);
        ASSERT_EQ(flattenXY(1, 0, 8), 8);

        dxySize dxySz;
        dxySz.depth = 0;
        dxySz.x = 4;
        dxySz.y = 8;
        ASSERT_EQ(flattenXY(2, 3, dxySz), 2*8 + 3);
    }

    {
        LOG("Save/restore weights, split convolution network");

        /*
                   /-20*32x32[7x7]--20*8x8[2x2]--\
            1*32x32                               1*8x8--1*4x4--1*10x1
                   \                             /
                    \-1*8x8[1x3]----------------/
         */

        string topologyConfig =
            "input size 32x32\n"
            "layerConv size 10*32x32 from input convolve 7x7\n"
            "layerPool size 5*8x8 from layerConv pool max 2x2\n"
            "layerMix size 8x8 from layerPool\n"
            "layerGauss size 8x8 from input radius 1x3 tf gaussian\n"
            "layerCombine size 4x4 from layerMix\n"
            "layerCombine size 4x4 from layerGauss\n"
            "output size 10 from layerCombine\n";

        string inputDataConfig =
            "../images/8x8-test.bmp 0 0 0 0 0 0 0 0 0 0\n";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        const string filename = "./unitTestSavedWeights.txt";

        Net myNet1(topologyConfigFilename, false);
        myNet1.sampleSet.loadSamples(inputDataConfigFilename);
        myNet1.feedForward(myNet1.sampleSet.samples[0]);
        myNet1.backProp(myNet1.sampleSet.samples[0]);
        myNet1.saveWeights(filename);

        Net myNet2(topologyConfigFilename, false);
        myNet2.loadWeights(filename);

        // input layer has no incoming weights

        // layerConv:
        auto const &conv1kernels = layerNamed(myNet1, "layerConv")->flatConvolveMatrix;
        auto const &conv2kernels = layerNamed(myNet2, "layerConv")->flatConvolveMatrix;
        for (uint32_t depthNum = 0; depthNum < conv1kernels.size(); ++depthNum) {
            auto const &kernel1Flat = conv1kernels[depthNum];
            auto const &kernel2Flat = conv2kernels[depthNum];
            for (uint32_t n = 0; n < kernel1Flat.size(); ++n) {
                ASSERT_FEQ(kernel1Flat[n], kernel2Flat[n]);
            }
        }

        // layerPool has no incoming weights

        // layerMix size 8x8 from layerPool
        auto const &mix1 = *layerNamed(myNet1, "layerMix");
        auto const &mix2 = *layerNamed(myNet2, "layerMix");
        for (uint32_t neuronNum = 0; neuronNum < mix1.neurons[0].size(); ++neuronNum) {
            auto const &neuron1 = mix1.neurons[0][neuronNum];
            auto const &neuron2 = mix2.neurons[0][neuronNum];
            for (uint32_t i = 0; i < neuron1.backConnectionsIndices.size(); ++i) {
                ASSERT_FEQ(myNet1.connections[neuron1.backConnectionsIndices[i]].weight,
                           myNet2.connections[neuron2.backConnectionsIndices[i]].weight);
            }
        }

        // layerGauss size 8x8 from input radius 1x3
        auto const &gauss1 = *layerNamed(myNet1, "layerGauss"); // myNet1.layers[3];
        auto const &gauss2 = *layerNamed(myNet2, "layerGauss"); // myNet2.layers[3];
        for (uint32_t neuronNum = 0; neuronNum < gauss1.neurons[0].size(); ++neuronNum) {
            auto const &neuron1 = gauss1.neurons[0][neuronNum];
            auto const &neuron2 = gauss2.neurons[0][neuronNum];
            for (uint32_t i = 0; i < neuron1.backConnectionsIndices.size(); ++i) {
                ASSERT_FEQ(myNet1.connections[neuron1.backConnectionsIndices[i]].weight,
                           myNet2.connections[neuron2.backConnectionsIndices[i]].weight);
            }
        }

        // layerCombine size 4x4
        auto const &combine1 = *layerNamed(myNet1, "layerCombine");
        auto const &combine2 = *layerNamed(myNet2, "layerCombine");
        for (uint32_t neuronNum = 0; neuronNum < combine1.neurons[0].size(); ++neuronNum) {
            auto const &neuron1 = combine1.neurons[0][neuronNum];
            auto const &neuron2 = combine2.neurons[0][neuronNum];
            for (uint32_t i = 0; i < neuron1.backConnectionsIndices.size(); ++i) {
                ASSERT_FEQ(myNet1.connections[neuron1.backConnectionsIndices[i]].weight,
                           myNet2.connections[neuron2.backConnectionsIndices[i]].weight);
            }
        }

        // output size 10
        auto const &output1 = *myNet1.layers.back();
        auto const &output2 = *myNet2.layers.back();
        for (uint32_t neuronNum = 0; neuronNum < output1.neurons[0].size(); ++neuronNum) {
            auto const &neuron1 = output1.neurons[0][neuronNum];
            auto const &neuron2 = output2.neurons[0][neuronNum];
            for (uint32_t i = 0; i < neuron1.backConnectionsIndices.size(); ++i) {
                ASSERT_FEQ(myNet1.connections[neuron1.backConnectionsIndices[i]].weight,
                           myNet2.connections[neuron2.backConnectionsIndices[i]].weight);
            }
        }
    }

    {
        // This test must be executed last, as it disrupts the Logger streams.
        // To do: fix that.
        LOG("test logger output");

        const string tempFilename = "./unitTestTempOutput";

        ofstream tmpFile;
        tmpFile.open(tempFilename);
        Net myNet("", false);
        info.pfile = &tmpFile;
        info << "HelloTestFile" << endl;
        tmpFile.flush();
        tmpFile.close();
        ifstream rFile(tempFilename);
        string result;
        rFile >> result;
        ASSERT_EQ(result, "HelloTestFile");
        rFile.close();

        tmpFile.open(tempFilename);
        err.pfile = &tmpFile;
        err << "HelloErrorTestFile" << endl;
        tmpFile.flush();
        tmpFile.close();
        rFile.open(tempFilename);
        rFile >> result;
        ASSERT_EQ(result, "HelloErrorTestFile");
        rFile.close();
    }
}


} // end of namespace NNet


int main()
{
    // Redirect the console output streams from neural2d so that they don't get
    // mixed with the unit test output:
    ofstream tmpFile("./unitTestOutputRedirect");
    NNet::info.pfile = &tmpFile;
    NNet::warn.pfile = &tmpFile;
    NNet::err.pfile = &tmpFile;

    try {
        NNet::unitTestConfigParsers();
        NNet::unitTestImages();
        NNet::unitTestNet();
        NNet::unitTestSparseConnections();
        NNet::unitTestConvolutionFiltering();
        NNet::unitTestConvolutionNetworking();
        NNet::unitTestPooling();
        NNet::unitTestMisc();
    } catch (...) {
        cerr << "Oops, something didn't work right." << endl;
        return 1;
    }

    if (NNet::numErrors == 0) {
        cout << "PASS: All tests passed." << endl;
    } else if (NNet::numErrors == 1){
        cout << "There was only one error." << endl;
    } else {
        cout << "There were " << NNet::numErrors << " errors." << endl;
    }

    return 0;
}
