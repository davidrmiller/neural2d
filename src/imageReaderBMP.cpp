/*
imageReaderBMP.cpp -- this is part of the neural2d program.
David R. Miller, 2014, 2015
https://github.com/davidrmiller/neural2d
Also see neural2d.h for more information.
*/

#include "neural2d.h"

namespace NNet {


// Extract the input data from the specified file and save the data in the data container.
// Returns the nonzero image size if successful, else returns 0,0.
//
xySize ImageReaderBMP::getData(std::string const &filename, std::vector<float> &dataContainer, ColorChannel_t colorChannel)
{
    FILE* f = fopen(filename.c_str(), "rb");

    if (f == NULL) {
        return { 0, 0 };
    }

    // Read the BMP header to get the image dimensions:

    unsigned char info[54];
    if (fread(info, sizeof(unsigned char), 54, f) != 54) {
        fclose(f);
        return { 0, 0 };
    }

    if (info[0] != 'B' || info[1] != 'M') {
        fclose(f);
        return { 0, 0 };
    }

    // Verify the offset to the pixel data. It should be the same size as the info[] data read above.

    size_t dataOffset = (info[13] << 24)
                      + (info[12] << 16)
                      + (info[11] << 8)
                      +  info[10];

    // Verify that the file contains 24 bits (3 bytes) per pixel (red, green blue at 8 bits each):

    int pixelDepth = (info[29] << 8) + info[28];
    if (pixelDepth != 24) {
        fclose(f);
        return { 0, 0 };
    }

    // This method of converting 4 bytes to a uint32_t is portable for little- or
    // big-endian environments:

    uint32_t width = (info[21] << 24)
                   + (info[20] << 16)
                   + (info[19] << 8)
                   +  info[18];

    uint32_t height = (info[25] << 24)
                    + (info[24] << 16)
                    + (info[23] << 8)
                    +  info[22];

    // Position the read pointer to the first byte of pixel data:

    if (fseek(f, dataOffset, SEEK_SET) != 0) {
        fclose(f);
        return { 0, 0 };
    }

    uint32_t rowLen_padded = (width*3 + 3) & (~3);
    std::unique_ptr<unsigned char[]> imageData {new unsigned char[rowLen_padded]};

    dataContainer.clear();
    dataContainer.assign(width * height, 0); // Pre-allocate to make random access easy

    // Fill the data container with 8-bit data taken from the image data:

    for (uint32_t y = 0; y < height; ++y) {
        if (fread(imageData.get(), sizeof(unsigned char), rowLen_padded, f) != rowLen_padded) {
            fclose(f);
            return { 0, 0 };
        }

        // BMP pixels are arranged in memory in the order (B, G, R):

        unsigned val = 0;

        for (uint32_t x = 0; x < width; ++x) {
            if (colorChannel == NNet::R) {
                val = imageData[x * 3 + 2]; // Red
            } else if (colorChannel == NNet::G) {
                val = imageData[x * 3 + 1]; // Green
            } else if (colorChannel == NNet::B) {
                val = imageData[x * 3 + 0]; // Blue
            } else if (colorChannel == NNet::BW) {
                // Rounds down:
                val = (unsigned)(0.3 * imageData[x*3 + 2] +   // Red
                                 0.6 * imageData[x*3 + 1] +   // Green
                                 0.1 * imageData[x*3 + 0]);   // Blue
            } else {
                err << "Error: unknown pixel conversion" << endl;
                throw exceptionImageFile();
            }

            // Convert the pixel from the range 0..256 to a smaller
            // range that we can input into the neural net:
            // Also we'll invert the rows so that the origin is the upper left at 0,0:

            dataContainer[flattenXY(x, (height - y) - 1, height)] = pixelToNetworkInputRange(val);
        }
    }

    fclose(f);

    return { width, height };
}

} // end namespace NNet
