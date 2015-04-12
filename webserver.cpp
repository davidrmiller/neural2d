/*
webserver.cpp -- embedded web server for the GUI for the neural2d program.
David R. Miller, 2014, 2015
For more information, see neural2d.h and https://github.com/davidrmiller/neural2d .
*/

#include <cassert>
#include <condition_variable> // For mutex
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>     // POSIX, for read(), write(), close()
#include <sys/types.h>  // For setsockopt() and SO_REUSEADDR
#include <sys/socket.h> // For setsockopt() and SO_REUSEADDR

#include "webserver.h"


namespace NNet {

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

    // Set SO_REUSEADDR so we can bind the port even if there are
    // connections in the TIME_WAIT state, thus allowing the webserver
    // to be restarted without waiting for the TIME_WAIT to expire.

    int optval = 1;
    setsockopt(socketFd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

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

} /* end namespace NNet */
