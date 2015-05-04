/*
unitTest.cpp for use with neural2d.
https://github.com/davidrmiller/neural2d
David R. Miller, 2015

To use these unit tests, build with the Makefile target "unitTest," then
execute in the same directory where you build neural2d. Everything must
be compiled with the webserver disabled (-DDISABLE_WEBSERVER).
*/

#include "neural2d.h"

using namespace std;

namespace NNet {

// Define to skip exception testing:
//#define SKIP_TEST_EXCEPTIONS

class unitTestException : std::exception { };
extern float pixelToNetworkInputRange(unsigned val);

// Unit tests use the following macros to log information and report problems.
// The only console output from a unit test should be the result of invoking
// these macros.

#define LOG(s) (cerr << "LOG: " << s << endl)

#define ASSERT_EQ(c, v) { if (!((c)==(v))) { \
    cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), expected " \
    << (v) << ", got " << (c) << endl; \
    throw unitTestException(); \
    } }

#define ASSERT_NE(c, v) { if (!((c)!=(v))) { \
    cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), got unexpected " \
    << (v) << endl; \
    throw unitTestException(); \
    } }

#define ASSERT_GE(c, v) { if (!((c)>=(v))) { \
    cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), expected >=" \
    << (v) << ", got " << (c) << endl; \
    throw unitTestException(); \
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
        caught = true; \
    } \
    if (!caught) { \
        cerr << "FAIL: in " << __FILE__ << "(" << __LINE__ << "), did not catch expected exception " \
             << endl; \
        throw unitTestException(); \
    } \
}


// For testing purposes, it's sometimes useful to force all the connection
// weights to a fixed value:
//
void setAllWeights(Net &myNet, float w)
{
    for (auto &conn : myNet.connections) {
        conn.weight = w;
    }
}

int unitTestConfigParser()
{
    LOG("unitTestConfigParser()");

    // Make an uninitialized Net:
    Net myNet("");

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
        ASSERT_EQ(specs[0].layerParams.channel, NNet::BW);
        ASSERT_EQ(specs[0].layerParams.convolveMatrix.size(), 0);
        ASSERT_EQ(specs[0].layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(specs[0].layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(specs[0].layerParams.layerName, "input");
        ASSERT_EQ(specs[0].layerParams.poolSize.x, 0);
        ASSERT_EQ(specs[0].layerParams.poolSize.y, 0);
        ASSERT_EQ(specs[0].layerParams.size.depth, 1);
        ASSERT_EQ(specs[0].layerParams.size.x, 2);
        ASSERT_EQ(specs[0].layerParams.size.y, 2);

        ASSERT_EQ(specs[1].fromLayerName, "input");
        ASSERT_EQ(specs[1].configLineNum, 2);
        ASSERT_EQ(specs[1].layerParams.convolveMatrix.size(), 0);
        ASSERT_EQ(specs[1].layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(specs[1].layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(specs[1].layerParams.layerName, "output");
        ASSERT_EQ(specs[1].layerParams.poolSize.x, 0);
        ASSERT_EQ(specs[1].layerParams.poolSize.y, 0);
        ASSERT_EQ(specs[1].layerParams.size.depth, 1);
        ASSERT_EQ(specs[1].layerParams.size.x, 1);
        ASSERT_EQ(specs[1].layerParams.size.y, 1);
        ASSERT_NE(specs[1].layerParams.tf, 0);
        ASSERT_NE(specs[1].layerParams.tfDerivative, 0);
        ASSERT_EQ(specs[1].layerParams.transferFunctionName.size(), sizeof "tanh" - 1);
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
        ASSERT_EQ(specs[0].layerParams.channel, NNet::R);
    }

    {
        LOG("Test dxySize: only X given");

        string config =
            "input size 3\n"
            "output from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs[0].layerParams.size.depth, 1);
        ASSERT_EQ(specs[0].layerParams.size.x, 3);
        ASSERT_EQ(specs[0].layerParams.size.y, 1);
    }

    {
        LOG("Test dxySize(): depth and X given");

        string config =
            "input size 4*3\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs[0].layerParams.size.depth, 4);
        ASSERT_EQ(specs[0].layerParams.size.x, 3);
        ASSERT_EQ(specs[0].layerParams.size.y, 1);
    }

    {
        LOG("Test dxySize: depth, X, and Y given");

        string config =
            "input size 4*3x5\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs[0].layerParams.size.depth, 4);
        ASSERT_EQ(specs[0].layerParams.size.x, 3);
        ASSERT_EQ(specs[0].layerParams.size.y, 5);
    }

    {
        LOG("Test whitespace tolerance");

        string config =
            "  input size 1\n"
            "output from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs.size(), 2);
        ASSERT_EQ(specs[0].layerParams.layerName, "input");
    }

    {
        LOG("Test whitespace tolerance 2");

        string config =
            "input size 4 * 5 x 6 \n"
            "output size 1\tfrom input\t\n\t";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs.size(), 2);
        ASSERT_EQ(specs[0].layerParams.size.depth, 4);
        ASSERT_EQ(specs[0].layerParams.size.x, 5);
        ASSERT_EQ(specs[0].layerParams.size.y, 6);
    }

    {
        LOG("from parameter");

        string config = \
            "input size 1\n" \
            "layer1 size 1 from input\n" \
            "layer2 size 2x2 from layer1\n" \
            "layer3 size 7x8 from input\n" \
            "layer4 size 2x2 from layer3\n" \
            "layer5 from input\n" \
            "layer6 from layer4\n" \
            "output size 1 from layer6\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        ASSERT_EQ(specs.size(), 8);
        auto *spec = &specs[0];
        ASSERT_EQ(spec->fromLayerName.size(), 0);
        ASSERT_EQ(spec->configLineNum, 1);
        ASSERT_EQ(spec->layerParams.channel, NNet::BW);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 0);
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.layerName, "input");
        ASSERT_EQ(spec->layerParams.poolSize.x, 0);
        ASSERT_EQ(spec->layerParams.poolSize.y, 0);
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 1);
        ASSERT_EQ(spec->layerParams.size.y, 1);

        spec = &specs[1];   // layer1 size 1 from input
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "input" - 1);
        ASSERT_EQ(spec->configLineNum, 2);
        ASSERT_EQ(spec->layerParams.channel, NNet::BW);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 0);
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.layerName, "layer1");
        ASSERT_EQ(spec->layerParams.poolSize.x, 0);
        ASSERT_EQ(spec->layerParams.poolSize.y, 0);
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 1);
        ASSERT_EQ(spec->layerParams.size.y, 1);
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName.size(), sizeof "tanh" - 1);

        spec = &specs[2];   // layer2 size 2x2 from layer1
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "layer1" - 1);
        ASSERT_EQ(spec->configLineNum, 3);
        ASSERT_EQ(spec->layerParams.channel, NNet::BW);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 0);
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.layerName, "layer2");
        ASSERT_EQ(spec->layerParams.poolSize.x, 0);
        ASSERT_EQ(spec->layerParams.poolSize.y, 0);
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 2);
        ASSERT_EQ(spec->layerParams.size.y, 2);
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName.size(), sizeof "tanh" - 1);

        spec = &specs[3];   // layer3 size 7x8 from input
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "input" - 1);
        ASSERT_EQ(spec->configLineNum, 4);
        ASSERT_EQ(spec->layerParams.channel, NNet::BW);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 0);
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.layerName, "layer3");
        ASSERT_EQ(spec->layerParams.poolSize.x, 0);
        ASSERT_EQ(spec->layerParams.poolSize.y, 0);
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 7);
        ASSERT_EQ(spec->layerParams.size.y, 8);
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName.size(), sizeof "tanh" - 1);

        spec = &specs[4];   // layer4 size 2x2 from layer3
        ASSERT_EQ(spec->fromLayerName.size(), sizeof "layer3" - 1);
        ASSERT_EQ(spec->configLineNum, 5);
        ASSERT_EQ(spec->layerParams.channel, NNet::BW);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 0);
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.layerName, "layer4");
        ASSERT_EQ(spec->layerParams.poolSize.x, 0);
        ASSERT_EQ(spec->layerParams.poolSize.y, 0);
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 2);
        ASSERT_EQ(spec->layerParams.size.y, 2);
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName.size(), sizeof "tanh" - 1);
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
        ASSERT_EQ(spec->layerParams.radius.x, 2);
        ASSERT_EQ(spec->layerParams.radius.y, 3);
        spec = &specs[2];
        ASSERT_EQ(spec->layerParams.radius.x, 4);
        ASSERT_EQ(spec->layerParams.radius.y, 1);
        spec = &specs[3];
        ASSERT_EQ(spec->layerParams.radius.x, 0);
        ASSERT_EQ(spec->layerParams.radius.y, 4);
        spec = &specs[4];
        ASSERT_GE(spec->layerParams.radius.x, Net::HUGE_RADIUS);
        ASSERT_GE(spec->layerParams.radius.y, Net::HUGE_RADIUS);
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
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName, "linear");
        spec = &specs[2];
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName, "gaussian");
        spec = &specs[3];
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName, "tanh");
        spec = &specs[4];
        ASSERT_NE(spec->layerParams.tf, 0);
        ASSERT_NE(spec->layerParams.tfDerivative, 0);
        ASSERT_EQ(spec->layerParams.transferFunctionName, "logistic");
    }

    {
        LOG("test that missing size matches from-layer");

        string config =
            "input size 2x3\n"
            "layer1 from input\n"
            "layer2 size 4x5 from input\n"
            "layer3 from layer2\n"
            "layer4 size 2*3x4 from input\n"
            "layer5 from layer4\n"
            "output size 1 from layer5\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        auto *spec = &specs[1];
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 2);
        ASSERT_EQ(spec->layerParams.size.y, 3);
        spec = &specs[3];
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 4);
        ASSERT_EQ(spec->layerParams.size.y, 5);
        spec = &specs[5];
        ASSERT_EQ(spec->layerParams.size.depth, 2);
        ASSERT_EQ(spec->layerParams.size.x, 3);
        ASSERT_EQ(spec->layerParams.size.y, 4);
        spec = &specs[6];
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 1);
        ASSERT_EQ(spec->layerParams.size.y, 1);
    }

    {
        LOG("test that missing size matches from-layer 2");

        string config =
            "input size 1\n"
            "output from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        auto *spec = &specs[1];
        ASSERT_EQ(spec->layerParams.size.depth, 1);
        ASSERT_EQ(spec->layerParams.size.x, 1);
        ASSERT_EQ(spec->layerParams.size.y, 1);
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
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, true);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 1);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0].size(), 1);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0].size(), 1);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0][0], 2);
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
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, true);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 1);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0].size(), 2);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0].size(), 1);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0][0], 2);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][1][0], 3);
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
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, true);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, false);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 1);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0].size(), 1);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0].size(), 2);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0][0], 2);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0][1], 3);
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
        ASSERT_EQ(spec->layerParams.size.depth, 10);
        ASSERT_EQ(spec->layerParams.size.x, 16);
        ASSERT_EQ(spec->layerParams.size.y, 16);
        ASSERT_EQ(spec->layerParams.kernelSize.x, 3);
        ASSERT_EQ(spec->layerParams.kernelSize.y, 4);
        ASSERT_EQ(spec->layerParams.isConvolutionFilterLayer, false);
        ASSERT_EQ(spec->layerParams.isConvolutionNetworkLayer, true);
        ASSERT_EQ(spec->layerParams.convolveMatrix.size(), 10);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0].size(), 3);
        ASSERT_EQ(spec->layerParams.convolveMatrix[0][0].size(), 4);
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
            "layer1 size 10*16x16 from input pool max radius 2x3\n"
            "output size 1 from input\n";

        istringstream ss(config);
        auto specs = myNet.parseTopologyConfig(ss);
        auto *spec = &specs[1];
        ASSERT_EQ(spec->layerParams.size.depth, 10);
        ASSERT_EQ(spec->layerParams.poolMethod, POOL_MAX);
        ASSERT_EQ(spec->layerParams.poolSize.x, 2);
        ASSERT_EQ(spec->layerParams.poolSize.y, 3);
    }


    return 0;
}


int unitTestNet()
{
    LOG("unitTestNet()");

    {
        LOG("Smoke test null config");
        Net myNet("");
    }

    {
        LOG("Smoke test trivial config");
        string config =
            "input size 1\n"
            "output from input\n";

        istringstream ss(config);
        Net myNet("");
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));
        ASSERT_EQ((int)(100*myNet.alpha + 0.5), 10); // Expecting 0.1 which is inexact in floating-point
        ASSERT_EQ(myNet.bias.output, 1.0);
        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0].neurons.size(), 1);
        ASSERT_EQ(myNet.layers[1].neurons.size(), 1);

        const Neuron *pNeuron = &myNet.layers[0].neurons[0];
        ASSERT_EQ(pNeuron->forwardConnectionsIndices.size(), 1);
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 0);

        pNeuron = &myNet.layers[1].neurons[0];
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
        Net myNet("");
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0].neurons.size(), 10*10);
        ASSERT_EQ(myNet.layers[1].neurons.size(), 8*6);

        const Neuron *pNeuron = &myNet.layers[0].neurons[0];
        ASSERT_EQ(pNeuron->forwardConnectionsIndices.size(), 8*6);
        ASSERT_EQ(pNeuron->backConnectionsIndices.size(), 0);

        ASSERT_EQ(myNet.connections.size(), 8*6*10*10 + 8*6);
    }

    {
        LOG("neuron connections");

        string topologyConfig =
            "input size 1\n"
            "output size 1 from input radius 0x0 tf linear\n"; // One-to-one

        const string topologyConfigFilename = "./topologyUnitTest.txt";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        Net myNet(topologyConfigFilename);

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0].neurons.size(), 1);
        ASSERT_EQ(myNet.layers[1].neurons.size(), 1);

        ASSERT_EQ(myNet.layers[0].params.isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[0].params.isConvolutionNetworkLayer, false);
        ASSERT_EQ(myNet.layers[1].params.isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[1].params.isConvolutionNetworkLayer, false);

        ASSERT_EQ(myNet.layers[0].params.size.depth, 1);
        ASSERT_EQ(myNet.layers[0].params.size.x, 1);
        ASSERT_EQ(myNet.layers[0].params.size.y, 1);

        ASSERT_EQ(myNet.layers[1].params.size.depth, 1);
        ASSERT_EQ(myNet.layers[1].params.size.x, 1);
        ASSERT_EQ(myNet.layers[1].params.size.y, 1);

        ASSERT_EQ(myNet.connections.size(), 1+1); // one plus bias
        ASSERT_EQ(myNet.layers[0].neurons[0].backConnectionsIndices.size(), 0);
        ASSERT_EQ(myNet.layers[0].neurons[0].forwardConnectionsIndices.size(), 1);
        ASSERT_EQ(myNet.layers[1].neurons[0].backConnectionsIndices.size(), 2); // Source plus bias
        ASSERT_EQ(myNet.layers[1].neurons[0].forwardConnectionsIndices.size(), 0);

        ASSERT_EQ(myNet.bias.backConnectionsIndices.size(), 0);
        ASSERT_EQ(myNet.bias.forwardConnectionsIndices.size(), 1);

        auto const &l0n0 = myNet.layers[0].neurons[0];
        auto const &l1n0 = myNet.layers[1].neurons[0];
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

        const string topologyConfigFilename = "./topologyUnitTest.txt";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        Net myNet(topologyConfigFilename);

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0].neurons.size(), 8*8);
        ASSERT_EQ(myNet.layers[1].neurons.size(), 8*8);

        ASSERT_EQ(myNet.layers[0].params.isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[0].params.isConvolutionNetworkLayer, false);
        ASSERT_EQ(myNet.layers[1].params.isConvolutionFilterLayer, false);
        ASSERT_EQ(myNet.layers[1].params.isConvolutionNetworkLayer, false);

        ASSERT_EQ(myNet.layers[0].params.size.depth, 1);
        ASSERT_EQ(myNet.layers[0].params.size.x, 8);
        ASSERT_EQ(myNet.layers[0].params.size.y, 8);

        ASSERT_EQ(myNet.layers[1].params.size.depth, 1);
        ASSERT_EQ(myNet.layers[1].params.size.x, 8);
        ASSERT_EQ(myNet.layers[1].params.size.y, 8);
    }

    return 0;
}


int unitTestConvolutionFilter()
{
    LOG("unitTestConvolutionFilter()");

    {
        LOG("radius parameter");
        string config =
            "input size 10x10\n"
            "output size 8x8 from input radius 0x0\n";

        istringstream ss(config);
        Net myNet("");
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0].neurons.size(), 10*10);
        ASSERT_EQ(myNet.layers[1].neurons.size(), 8*8);

        const Neuron *pNeuron = &myNet.layers[0].neurons[0];
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
        Net myNet("");
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0].neurons.size(), 10*10);
        ASSERT_EQ(myNet.layers[1].neurons.size(), 1);

        const Neuron *pNeuron = &myNet.layers[0].neurons[0];
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
        Net myNet("");
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));

        ASSERT_EQ(myNet.layers.size(), 2);
        ASSERT_EQ(myNet.layers[0].neurons.size(), 10*10);
        ASSERT_EQ(myNet.layers[1].neurons.size(), 1);
        ASSERT_EQ(myNet.connections.size(), 3 + 1);
    }

    {
        LOG("radius parameter 4");
        string config =
            "input size 10x10\n"
            "output size 1 from input radius 1x1\n";

        istringstream ss(config);
        Net myNet("");
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));
        ASSERT_EQ(myNet.connections.size(), 5 + 1); // Assumes elliptical projection
    }

    {
        LOG("radius parameter rectangular projection");
        string config =
            "input size 10x10\n"
            "output size 1 from input radius 1x1\n";

        istringstream ss(config);
        Net myNet("");
        myNet.projectRectangular = true;
        myNet.configureNetwork(myNet.parseTopologyConfig(ss));
        ASSERT_EQ(myNet.connections.size(), 9 + 1); // Assumes elliptical projection
    }

    {
        LOG("Convolution kernel radius 1x0");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output size 8x8 from input radius 1x0 tf linear\n"; // 3 cols, one row

        string inputDataConfig =
            "images/8x8-test11.bmp\n";

        const string topologyConfigFilename = "./topologyUnitTest.txt";
        const string inputDataConfigFilename = "./inputDataUnitTest.txt";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);
        ASSERT_EQ(data[3], pixelToNetworkInputRange(8));
        ASSERT_EQ(data[4], pixelToNetworkInputRange(8));
        ASSERT_EQ(data[5], pixelToNetworkInputRange(8));

        myNet.feedForward(myNet.sampleSet.samples[0]);
        auto &inputLayer = myNet.layers[0];
        auto &outputLayer = myNet.layers.back();

        // Neuron 4 is on a row of all 8's, so its output should be the sum
        // of three pixels of value 8 plus a bias:
        ASSERT_EQ(inputLayer.neurons[3].output, pixelToNetworkInputRange(8));
        ASSERT_EQ(inputLayer.neurons[4].output, pixelToNetworkInputRange(8));
        ASSERT_EQ(inputLayer.neurons[5].output, pixelToNetworkInputRange(8));
        ASSERT_EQ(outputLayer.neurons[4].output, 3 * pixelToNetworkInputRange(8) + 1.0);
    }

    {
        LOG("Convolution kernel radius 0x1");

        string topologyConfig =
            "input size 8x8 channel G\n"
            "output size 8x8 from input radius 0x1 tf linear\n"; // One col, 3 rows

        string inputDataConfig =
            "images/8x8-test11.bmp\n";

        const string topologyConfigFilename = "./topologyUnitTest.txt";
        const string inputDataConfigFilename = "./inputDataUnitTest.txt";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename);
        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::G);
        myNet.feedForward(myNet.sampleSet.samples[0]);

        // Neuron 8 is on a col of all 1's, so its output should be the sum
        // of three pixels of value 1 plus a bias:
        ASSERT_EQ(myNet.layers.back().neurons[8].output, 3 * pixelToNetworkInputRange(1) + 1.0);

        // Neuron 15 is on a col of all 8's, so its output should be the sum
        // of three pixels of value 8 plus a bias:
        ASSERT_EQ(myNet.layers.back().neurons[15].output, 3 * pixelToNetworkInputRange(8) + 1.0);
    }

    return 0;
}


int unitTestImages()
{
    {
        LOG("images smoke test");

        // to do: We should verify here that the topology.txt and inputData.txt
        // files are the default version, and that the digits archive has been
        // expanded in the usual subdirectory

        Net myNet("topology.txt");
        ASSERT_EQ(myNet.layers.size(), 3);
        ASSERT_EQ(myNet.layers[0].params.size.x, 32);
        ASSERT_EQ(myNet.layers[0].params.size.y, 32);
        ASSERT_EQ(myNet.sampleSet.samples.size(), 0);

        myNet.sampleSet.loadSamples("./inputData.txt");
        ASSERT_EQ(myNet.sampleSet.samples.size(), 5000);
    }

    {
        LOG("8x8-test.bmp orientation");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output size 8x8 from input radius 0x0\n";

        string inputDataConfig =
            "images/8x8-test.bmp\n";

        const string topologyConfigFilename = "./topologyUnitTest.txt";
        const string inputDataConfigFilename = "./inputDataUnitTest.txt";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename);
        ASSERT_EQ(myNet.connections.size(), 8*8 + 8*8);

        setAllWeights(myNet, 1.0);
        ASSERT_EQ((int)(myNet.connections[0].weight * 100 + 0.5), 100);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        ASSERT_EQ(myNet.sampleSet.samples.size(), 1);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);
        ASSERT_EQ(data.size(), 8*8);

        ASSERT_EQ(data[0],   pixelToNetworkInputRange(30));
        ASSERT_EQ(data[1],   pixelToNetworkInputRange(101));
        ASSERT_EQ(data[7],   pixelToNetworkInputRange(40));
        ASSERT_EQ(data[63],  pixelToNetworkInputRange(20));
    }

    {
        LOG("8x8-test11.bmp channels");

        string topologyConfig =
            "input size 8x8 channel R\n"
            "output size 1 from input tf linear\n"; // With weights=1, this will sum the inputs+bias

        string inputDataConfig =
            "images/8x8-test11.bmp\n";

        const string topologyConfigFilename = "./topologyUnitTest.txt";
        const string inputDataConfigFilename = "./inputDataUnitTest.txt";

        std::ofstream topologyConfigFile(topologyConfigFilename);
        topologyConfigFile << topologyConfig;
        topologyConfigFile.close();

        std::ofstream inputDataConfigFile(inputDataConfigFilename);
        inputDataConfigFile << inputDataConfig;
        inputDataConfigFile.close();

        Net myNet(topologyConfigFilename);
        ASSERT_EQ(myNet.connections.size(), 8*8 + 1);

        setAllWeights(myNet, 1.0);

        myNet.sampleSet.loadSamples(inputDataConfigFilename);
        auto data = myNet.sampleSet.samples[0].getData(NNet::R);

        ASSERT_EQ(data[0],  pixelToNetworkInputRange(8));
        ASSERT_EQ(data[1],  pixelToNetworkInputRange(8));
        ASSERT_EQ(data[8],  pixelToNetworkInputRange(7));
        ASSERT_EQ(data[63], pixelToNetworkInputRange(1));

        myNet.feedForward(myNet.sampleSet.samples[0]);
        // Output should be the sum of 64 input values of pixels with values from 1..8.
        //   plus a bias input of 1.0:
        float avgInputVal = (pixelToNetworkInputRange(1) + pixelToNetworkInputRange(8)) / 2.0;
        ASSERT_EQ(myNet.layers.back().neurons[0].output, 64 * avgInputVal + 1.0);

        myNet.sampleSet.clearImageCache();
        data = myNet.sampleSet.samples[0].getData(NNet::R);
        ASSERT_EQ(data[0],  pixelToNetworkInputRange(8));
        ASSERT_EQ(data[1],  pixelToNetworkInputRange(8));
        ASSERT_EQ(data[8],  pixelToNetworkInputRange(7));
        ASSERT_EQ(data[63], pixelToNetworkInputRange(1));

        myNet.sampleSet.clearImageCache();
        data = myNet.sampleSet.samples[0].getData(NNet::G);
        ASSERT_EQ(data[0],  pixelToNetworkInputRange(1));
        ASSERT_EQ(data[1],  pixelToNetworkInputRange(2));
        ASSERT_EQ(data[8],  pixelToNetworkInputRange(1));
        ASSERT_EQ(data[63], pixelToNetworkInputRange(8));

        myNet.sampleSet.clearImageCache();
        data = myNet.sampleSet.samples[0].getData(NNet::B);
        ASSERT_EQ(data[0], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[1], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[8], pixelToNetworkInputRange(127));
        ASSERT_EQ(data[63],pixelToNetworkInputRange(127));
    }

    return 0;
}


int unitTestMisc()
{
    {
        LOG("test logger output");

        const string tempFilename = "./unitTestTempOutput";

        ofstream tmpFile;
        tmpFile.open(tempFilename);
        Net myNet("");
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

    return 0;
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
        NNet::unitTestConfigParser();
        NNet::unitTestNet();
        NNet::unitTestConvolutionFilter();
        NNet::unitTestImages();
        NNet::unitTestMisc();
    } catch (...) {
        cerr << "Oops, something didn't work right." << endl;
        return 1;
    }

    cout << "PASS: All tests passed." << endl;

    return 0;
}
