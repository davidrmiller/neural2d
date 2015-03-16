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

namespace NNet {

// A Thread-safe FIFO; pushes to the back, pops from the front. Push and
// pop are always non-blocking. If the queue is empty, pop() immediately
// returns with s set to an empty string.

struct Message_t
{
    Message_t(void) { text = ""; httpResponseFileDes = -1; };
    string text;
    int httpResponseFileDes;
};

class MessageQueue
{
public:
    MessageQueue() { };
    void push(Message_t &msg);
    void pop(Message_t &msg);
    MessageQueue(const MessageQueue &) = delete;            // No copying
    MessageQueue &operator=(const MessageQueue &) = delete; // No assignment

private:
    queue<Message_t> mqueue;
    mutex mmutex;
};


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

