/*
neural2d.cpp
David R. Miller, 2014
For more info, see neural2d.h.
For more info, see the tutorial video.
*/

#include "neural2d.h"

using namespace std;


int main(int argc, char **argv)
{
    // We need two or three filenames -- we can define them here, or get them from
    // the command line. If they are specified on the command line, they must be in
    // the order: topology, input-data, and optionally, weights.

    string topologyFilename;  // Always needed
    string inputDataFilename; // Always needed
    string weightsFilename;   // Needed only if saving or restoring weights

    if (argc > 1) topologyFilename  = argv[1];
    if (argc > 2) inputDataFilename = argv[2];
    if (argc > 3) weightsFilename   = argv[3];

    NNet::Net myNet(topologyFilename);   // Create net, neurons, and connections
    myNet.enableRemoteInterface = true;

    NNet::SampleSet mySamples(inputDataFilename);

    // Here is an example of TRAINING mode -------------:

    myNet.eta = 0.1;
    myNet.dynamicEtaAdjust = false;
    myNet.alpha = 0.001;
    myNet.lambda = 0.0;
    myNet.doneErrorThreshold = 0.5;
    myNet.reportEveryNth = 25;
    double doneErrorThreshold = 0.005;

    while (myNet.repeatInputSamples) {
        if (myNet.shuffleInputSamples) {
            mySamples.shuffle();
        }
        for (uint32_t sampleIdx = 0; sampleIdx < mySamples.samples.size(); ++sampleIdx) {
            NNet::Sample &sample = mySamples.samples[sampleIdx];
            myNet.feedForwardNew(sample);
            myNet.backProp(sample);
            myNet.reportResultsNew(sample);
            if (myNet.recentAverageError < doneErrorThreshold) {
                cout << "Solved!   -- Saving weights..." << endl;
                myNet.saveWeights(weightsFilename);
                sleep(5); // Do whatever else needs to be done here
                exit(0);
            }
        }
    }

    cout << "Done." << endl;
    return 0;
}
