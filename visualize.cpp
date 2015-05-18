/*
visualize.cpp, this file contains visualizer-related functions for neural2d's optional GUI
David R. Miller, 2015
https://github.com/davidrmiller/neural2d

Also see neural2d.h for more information.
*/

#include <memory>   // for unique_ptr
#include <string>
#include <vector>

#include "neural2d.h"

#if defined(ENABLE_WEBSERVER) && !defined(DISABLE_WEBSERVER)

namespace NNet {

extern unsigned networkInputValToPixelRange(float);


// base64Encode() is adapted from the public domain version found on
// http://en.wikibooks.org/wiki/Algorithm_Implementation/Miscellaneous/Base64

const static char base64charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const static char padCharacter = '=';

std::basic_string<char> base64Encode(std::vector<uint8_t> inputBuffer)
{
    std::basic_string<char> encodedString;
    encodedString.reserve(((inputBuffer.size() / 3) + (inputBuffer.size() % 3 > 0)) * 4);
    int64_t temp;
    auto cursor = inputBuffer.begin();

    for (size_t idx = 0; idx < inputBuffer.size() / 3; idx++) {
        temp  = (*cursor++) << 16; // Force big endian
        temp += (*cursor++) << 8;
        temp += (*cursor++);
        encodedString.append(1, base64charset[(temp & 0x00FC0000) >> 18]);
        encodedString.append(1, base64charset[(temp & 0x0003F000) >> 12]);
        encodedString.append(1, base64charset[(temp & 0x00000FC0) >> 6 ]);
        encodedString.append(1, base64charset[(temp & 0x0000003F)      ]);
    }

    switch(inputBuffer.size() % 3) {
    case 1:
        temp = (*cursor++) << 16; // Force big endian
        encodedString.append(1, base64charset[(temp & 0x00FC0000) >> 18]);
        encodedString.append(1, base64charset[(temp & 0x0003F000) >> 12]);
        encodedString.append(2, padCharacter);
        break;

    case 2:
        temp  = (*cursor++) << 16; // Force big endian
        temp += (*cursor++) << 8;
        encodedString.append(1, base64charset[(temp & 0x00FC0000) >> 18]);
        encodedString.append(1, base64charset[(temp & 0x0003F000) >> 12]);
        encodedString.append(1, base64charset[(temp & 0x00000FC0) >> 6 ]);
        encodedString.append(1, padCharacter);
        break;
    }

    return encodedString;
}


// Creates a 24-bit, 3-channel BMP image. The result is stored linearly in a vector container.
// R, G, and B channels are set identically.
//
std::shared_ptr<std::vector<uint8_t>> createBMPImage(uint8_t *pData, size_t width, size_t height)
{
    size_t padSize  = (4 - (width * 3) % 4) % 4;
    size_t sizeData = width * height * 3 + height * padSize;
    size_t sizeAll  = sizeData + 54;

    std::shared_ptr<std::vector<uint8_t>> pBmp{new std::vector<uint8_t>{
        // file header:
        'B', 'M',                  // Magic cookie

        (uint8_t)(sizeAll      ),  // Total size
        (uint8_t)(sizeAll >>  8),
        (uint8_t)(sizeAll >> 16),
        (uint8_t)(sizeAll >> 24),

        0, 0,
        0, 0,
        54, 0, 0, 0,               // Offset to data

        // info header:
        40, 0, 0, 0,               // Info header size

        (uint8_t)(width     ),     // Image width
        (uint8_t)(width >>  8),
        (uint8_t)(width >> 16),
        (uint8_t)(width >> 24),

        (uint8_t)(height      ),   // Image heigth
        (uint8_t)(height >>  8),
        (uint8_t)(height >> 16),
        (uint8_t)(height >> 24),

        255, 0,                    // Number of colors
        24, 0,                     // bpp
        0, 0, 0, 0,

        (uint8_t)(sizeData      ), // Unpadded image size
        (uint8_t)(sizeData >>  8),
        (uint8_t)(sizeData >> 16),
        (uint8_t)(sizeData >> 24),

        0x13, 0x0B, 0, 0,
        0x13, 0x0B, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    }};

    assert(pBmp->size() == 54);

    pBmp->reserve(pBmp->size() + sizeData);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            long red = pData[(height - y) * width + x]; // Invert rows

            pBmp->push_back(red);  // B channel
            pBmp->push_back(red);  // G channel
            pBmp->push_back(red);  // R channel
        }

        for (size_t i = 0; i < padSize; ++i) {
            pBmp->push_back((uint8_t)0);
        }
    }

    return pBmp;
}


// Returns a string containing menu options for the drop-down in the GUI
//
std::string Layer::visualizationsAvailable(void)
{
    return "";
}

std::string Layer::visualizeKernels()
{
    return "";
}

// Returns a base64-encoded BMP image
//
std::string Layer::visualizeOutputs()
{
    size_t numRows = size.y * size.depth;
    numRows += size.depth - 1; // Allow an extra row of pixels between images

    std::vector<uint8_t> image(size.x * numRows);
    auto it = begin(image);

    for (uint32_t depth = 0; depth < size.depth; ++depth) {
        for (uint32_t y = 0; y < size.y; ++y) {
            for (uint32_t x = 0; x < size.x; ++x) {
                auto const &neuron = neurons[depth][flattenXY(x, y, size)];
                *it++ = (uint8_t)(networkInputValToPixelRange(neuron.output));
            }
        }

        // Add a one-pixel row of white after every image except the last:
        if (depth != size.depth - 1) {
            for (uint32_t x = 0; x < size.x; ++x) {
                *it++ = (uint8_t)(255);
            }
        }
    }

    auto pBmp = createBMPImage(&image.front(), size.x, numRows);
    return base64Encode(*pBmp); // Return value optimization
}

std::string LayerConvolution::visualizationsAvailable(void)
{
    std::string menu = "";

    if (kernelSize.x >= 3 && kernelSize.y >= 3) {
        menu.append(", \"" + layerName + " kernels\"");
    }

    if (size.x >= 3 && size.y >= 3) {
        menu.append(", \"" + layerName + " activations\"");
    }

    return menu;
}

// Returns a base64-encoded BMP image
//
std::string LayerConvolution::visualizeKernels(void)
{
    size_t numRows = kernelSize.y * size.depth;
    numRows += size.depth - 1; // Allow an extra divider row between images

    std::vector<uint8_t> image(kernelSize.x * numRows);
    auto it = begin(image);

    for (uint32_t depth = 0; depth < size.depth; ++depth) {
        for (uint32_t y = 0; y < kernelSize.y; ++y) {
            for (uint32_t x = 0; x < kernelSize.x; ++x) {
                auto weight = flatConvolveMatrix[depth][flattenXY(x, y, kernelSize.y)];
                // Convert the floating point weight to a pixel value in the range 0..255:
                // This is one possible conversion; edit as needed:
                if (weight < 0.0) weight = -1.0;
                if (weight > 1.0) weight = 1.0;
                uint8_t pixel = networkInputValToPixelRange(weight);
                *it++ = pixel;
            }
        }

        // Add a one-pixel row of white except after the last image:
        if (depth != size.depth - 1) {
            for (uint32_t x = 0; x < kernelSize.x; ++x) {
                *it++ = (uint8_t)(255);
            }
        }
    }

    auto pBmp = createBMPImage(&image.front(), kernelSize.x, numRows);

    return base64Encode(*pBmp); // Return value optimization
}

std::string LayerPooling::visualizationsAvailable(void)
{
    if (size.x >= 3 && size.y >= 3) {
        return ", \"" + layerName + " activations\"";
    }

    return "";
}

std::string LayerRegular::visualizationsAvailable(void)
{
    if (layerName == "input") {
        return ", \"input layer\"";
    } else if (size.x >= 3 && size.y >= 3) {
        return ", \"" + layerName + " activations\"";
    }

    return "";
}

} // end using namespace NNet
#endif // end if webserver enabled
