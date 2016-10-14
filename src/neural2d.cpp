/*
neural2d.cpp
https://github.com/davidrmiller/neural2d
David R. Miller, 2014
For more info, see neural2d.h.
*/

#include "neural2d.h"

enum MODE { TRAINING, VALIDATE, TRAINED };

void printUsage(){
    std::cerr << "Usage is [--mode {TRAINING|VALIDATION|TRAINED}] [--weights <weightsfile>] [--p] <topologyfile> <inputdatafile>\n";
}

int main(int argc, char **argv)
{
    // We need two or three filenames -- we can define them here, or get them from
    // the command line. If they are specified on the command line, they must be in
    // the order: topology, input-data, and optionally, weights.

    std::string topologyFilename = "";   // Always needed
    std::string inputDataFilename = ""; // Always needed
    std::string weightsFilename = "weights.txt";     // Needed only if saving or restoring weights
    MODE mode = TRAINING;
    bool paused(false);

    if (argc < 3) { // Check the value of argc. If not enough parameters have been passed, inform user and exit.
        printUsage();
        return 0;
    } else { // if we got enough parameters...
        char* myFile, myPath, myOutPath;
        std::cout << argv[0];
        for (int i = 1; i < argc; i++) {
            if (i + 1 != argc){ // Check that we haven't finished parsing already
                if (std::string(argv[i]) == "--mode") {
                    std::string modestr = argv[++i];
                    if (modestr == "VALIDATE"){
                        mode = MODE::VALIDATE;
                    } else if (modestr == "TRAINED"){
                        mode = MODE::TRAINED;
                    }//else leave as default ... TRAINING
                } else if (std::string(argv[i]) == "--help") {
                    printUsage();
                } else if (std::string(argv[i]) == "--p") {
                    paused = true;
                    std::cout << "Paused." << std::endl;
                } else if (std::string(argv[i]) == "--weights") {
                    weightsFilename = argv[++i];
                } else {
                    if (topologyFilename == ""){
                        topologyFilename = argv[i];
                    } else if (inputDataFilename == ""){
                        inputDataFilename = argv[i];
                    } else {
                        std::cerr << "Invalid parameter: " << argv[i] << "\n";
                        return 1;
                    }
                }
            }
        }
    }

    NNet::Net myNet(topologyFilename);   // Create net, neurons, and connections
    myNet.sampleSet.loadSamples(inputDataFilename);
    if(paused){
        myNet.isRunning = false;
    }

    switch(mode){
        case TRAINED:
        case VALIDATE:
            // Here is an example of VALIDATE mode -------------:

            myNet.reportEveryNth = 1;
            myNet.repeatInputSamples = false;

            myNet.loadWeights(weightsFilename); // Use weights from a trained net

            do {
                for (auto &sample : myNet.sampleSet.samples) {
                    myNet.feedForward(sample);
                    myNet.reportResults(sample);
                }
            } while (myNet.repeatInputSamples);
            break;

        case TRAINING:
        default:
            // Here is an example of TRAINING mode -------------:

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
            break;
    }

    std::cout << "Done." << std::endl;

    return 0;
}

