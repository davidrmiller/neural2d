/*
neural2d.cpp
https://github.com/davidrmiller/neural2d
David R. Miller, 2014
For more info, see neural2d.h.
*/

#include "neural2d.h"

int main(int argc, char **argv)
{
    // We need two or three filenames -- we can define them here, or get them from
    // the command line. If they are specified on the command line, they must be in
    // the order: topology, input-data, and optionally, weights.

    std::string topologyFilename = "topology.txt";   // Always needed
    std::string inputDataFilename = "inputData.txt"; // Always needed
    std::string weightsFilename = "weights.txt";     // Needed only if saving or restoring weights

    if (argc > 1) topologyFilename  = argv[1];
    if (argc > 2) inputDataFilename = argv[2];
    if (argc > 3) weightsFilename   = argv[3];


    NNet::Net myNet(topologyFilename);   // Create net, neurons, and connections
    myNet.sampleSet.loadSamples(inputDataFilename);

    if (argc > 4 && argv[4][0] == '-' && argv[4][1] == 'p') {
        myNet.isRunning = false;
        std::cout << "Paused." << std::endl;
    }

    // Here is an example of TRAINING mode -------------:
    // See the GitHub wiki for example code for VALIDATE and TRAINED modes:
    // https://github.com/davidrmiller/neural2d/wiki

    myNet.eta = 0.1f;
    myNet.dynamicEtaAdjust = true;
    myNet.alpha = 0.0f;
    myNet.reportEveryNth = 1;
    myNet.repeatInputSamples = true;
    myNet.shuffleInputSamples = true;
    myNet.doneErrorThreshold = 0.01f;

    do {
        if (myNet.shuffleInputSamples) {
            myNet.sampleSet.shuffle();
        }

        for (auto &sample : myNet.sampleSet.samples) {
            myNet.feedForward(sample);
            myNet.backProp(sample);
            myNet.reportResults(sample);
            if (myNet.recentAverageError < myNet.doneErrorThreshold) {
                std::cout << "Solved!   -- Saving weights..." << std::endl;
                myNet.saveWeights(weightsFilename);
                exit(0);
            }
        }
    } while (myNet.repeatInputSamples);

    std::cout << "Done." << std::endl;

    return 0;
}

