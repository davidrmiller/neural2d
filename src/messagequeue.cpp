#include "messagequeue.h"

// Message queue
// A Thread-safe non-blocking FIFO; pushes to the back, pops from the front.
// If the queue is empty, pop() immediately returns with msg set to an empty string.

void NNet::MessageQueue::push(Message_t &msg)
{
    std::unique_lock<std::mutex> locker(mmutex);
    mqueue.push(msg);
}

void NNet::MessageQueue::pop(Message_t &msg)
{
    std::unique_lock<std::mutex> locker(mmutex);
    if (mqueue.empty()) {
        msg.text = "";
        msg.httpResponseFileDes = -1;
    } else {
        msg = mqueue.front();
        mqueue.pop();
    }
}

