/*
imageReaderDat.cpp -- this supports the .dat binary input file format for the neural2d program.
David R. Miller, 2014, 2015
https://github.com/davidrmiller/neural2d
Also see neural2d.h for more information.
*/

#include "neural2d.h"

namespace NNet {

// Reverses the byte order if needed in the current environment:
//
template <class T>
T fixEndianness(T *pN)
{
    uint8_t test = 0x0001;
    if (*(uint8_t *)&test == 0) { // This is a very fast test
        return *pN;
    } else {
        uint8_t *p = (uint8_t *)pN;
        std::reverse(p, p + sizeof(T));
        return *pN;
    }
}

// Fields in datHeader are big-endian.
// Floating point data starts at the specified offset and is big-endian.
// The purpose of the offsetToData member is to allow the header to expand to
// provide space for arbitrary user-defined data, which neural2d will ignore.
//
struct datHeader
{
    uint32_t magic;             // 0x6c89f6ad
    uint32_t width;             // > 1
    uint32_t height;            // > 1
    uint32_t numChannels;       // >= 1
    uint32_t bytesPerElement;   // 4 (single) or 8 (double) precision
    uint32_t offsetToData;      // >= sizeof(datHeader)
};


// Extract the input data from the specified file and save the data in the data container.
// Returns the nonzero image size if successful, else returns 0,0.
//
xySize ImageReaderDat::getData(std::string const &filename,
            std::vector<float> &dataContainer, ColorChannel_t colorChannel)
{
    datHeader hdr;

    std::ifstream f(filename, std::ios::binary);

    f.read((char *)&hdr, sizeof(hdr));
    if (f.gcount() != sizeof(datHeader)) {
        return { 0, 0 };
    }

    if (fixEndianness(&hdr.magic) != 0x6c89f6ad) {
        return { 0, 0 };
    }

    // Fix the endianness of the other 32-bit header fields:
    for (uint32_t *p32 = (uint32_t *)&hdr + 1;
                p32 < (uint32_t *)&hdr + sizeof(hdr) / sizeof(uint32_t); ++p32) {
        fixEndianness(p32);
    }

    if (hdr.width == 0 || hdr.height == 0 || hdr.offsetToData < sizeof(hdr)) {
        return { 0, 0 };
    }

    // Map the color channel enumeration to a channel index:
    uint32_t colorChannelNumber;
    switch (colorChannel) {
    case NNet::R: colorChannelNumber = 0; break;
    case NNet::G: colorChannelNumber = 1; break;
    case NNet::B: colorChannelNumber = 2; break;
    default:
        err << "Error: unsupported color channel specified for " << filename << std::endl;
        //return { 0, 0 };
        throw exceptionInputSamplesFile();
    }

    if (colorChannelNumber >= hdr.numChannels) {
        err << "The color channel specified for " << filename << " does not exist" << std::endl;
        //return { 0, 0 };
        throw exceptionInputSamplesFile();
    }

    // Position the stream at the start of the image data:
    f.seekg(hdr.offsetToData + colorChannelNumber * hdr.bytesPerElement * hdr.width * hdr.height);

    // Clear dataContainer and reserve enough space:
    dataContainer.clear();
    dataContainer.assign(hdr.width * hdr.height, 0.0);

    if (hdr.bytesPerElement == sizeof(float)) {
        for (uint32_t y = 0; y < hdr.height; ++y) {
            for (uint32_t x = 0; x < hdr.width; ++x) {
                float n;
                f.read((char *)&n, sizeof n);
                fixEndianness(&n);
                dataContainer[flattenXY(x, y, hdr.height)] = n;
            }
        }
    } else if (hdr.bytesPerElement == sizeof(double)) {
        for (uint32_t y = 0; y < hdr.height; ++y) {
            for (uint32_t x = 0; x < hdr.width; ++x) {
                double n;
                f.read((char *)&n, sizeof n);
                fixEndianness(&n);
                dataContainer[flattenXY(x, y, hdr.height)] = (float)n;
            }
        }
    } else {
        err << "In " << filename << ", " << hdr.bytesPerElement
            << " bytes per element is not supported." << std::endl;
        throw exceptionInputSamplesFile();
    }

    return { hdr.width, hdr.height };
}

} // end namespace NNet
