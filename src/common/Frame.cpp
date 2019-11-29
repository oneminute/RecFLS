#include "Frame.h"

class FrameData : public QSharedData
{
public:

};

Frame::Frame(QObject *parent) : QObject(parent), data(new FrameData)
{

}

Frame::Frame(const Frame &rhs) : data(rhs.data)
{

}

Frame &Frame::operator=(const Frame &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

Frame::~Frame()
{

}
