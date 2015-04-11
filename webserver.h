/*
webserver.h -- this is the embedded web server for the neural2d program.
David R. Miller, 2014
For more info, see neural2d.h and https://github.com/davidrmiller/neural2d
*/

#ifndef NEURAL_WEBSERVER
#define NEURAL_WEBSERVER

#if defined(__CYGWIN__)
// On Windows, cygwin is missing to_string(), so we'll make one here:
#include <sstream>
    template <typename T>
    std::string to_string(T value)
    {
      std::ostringstream os;
      os << value;
      return os.str();
    }
#endif

#include <condition_variable> // For mutex
#include <queue>
#include <string>

using namespace std;

// For web server:
#include <sys/socket.h>
#include <netinet/in.h>

#include "messagequeue.h"

namespace NNet {

class WebServer
{
public:
    WebServer(void);
    ~WebServer(void);
    void start(int portNumber, MessageQueue &messages);
    void stopServer(void);
    void sendHttpResponse(string parameterBlock, int httpResponseFileDes);
    void webServerThread(int portNumber, MessageQueue &messageQueue);
    int portNumber;
    int socketFd;

private:
    void initializeHttpResponse(void);
    void extractAndQueueMessage(string s, int httpConnectionFd, MessageQueue &messages);
    void replyToUnknownRequest(int httpConnectionFd);

    bool firstAccess;  // So that we can do something different on the first HTTP request
    string firstPart;  // First part of the HTTP response
    string secondPart; // Last part of the HTTP response
};

}

#endif /* end #ifndef NEURAL_WEBSERVER */

