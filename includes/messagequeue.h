#ifndef MESSAGEQUEUE_H
#define MESSAGEQUEUE_H

#include <mutex>
#include <queue>
#include <string>

namespace NNet {

// A Thread-safe FIFO; pushes to the back, pops from the front. Push and
// pop are always non-blocking. If the queue is empty, pop() immediately
// returns with s set to an empty string.

struct Message_t
{
    Message_t() { text = ""; httpResponseFileDes = -1; };
    std::string text;
    int httpResponseFileDes;
};

class MessageQueue {
 public:
  MessageQueue() { };
  
  void push(Message_t &msg);
  void pop(Message_t &msg);
  
  MessageQueue(const MessageQueue &) = delete;            // No copying
  MessageQueue &operator=(const MessageQueue &) = delete; // No assignment

 private:
  std::queue<Message_t> mqueue;
  std::mutex mmutex;
};

}

#endif//MESSAGEQUEUE_H
