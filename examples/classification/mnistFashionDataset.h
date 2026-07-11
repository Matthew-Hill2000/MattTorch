#pragma once

#include <mattTorch/mattTorch.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <plotlypp/figure.hpp>
#include <plotlypp/traces/heatmap.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mattTorch/dataset/dataset.h"

namespace mattTorch {

class MNISTFashionDataset : public dataset {
 private:
  static constexpr int numClasses = 10;

  std::array<std::string_view, numClasses> classNames{
      "t-shirt", "trouser", "pullover", "dress", "coat",
      "sandal",  "shirt",   "sneaker",  "bag",   "ankle boot"};

  std::string datasetDir = "examples/classification/dataset";

  std::vector<std::string> filenames{
      "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
      "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"};

  std::string url =
      "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com";

  const std::string& imageFile(std::string_view type) const {
    return type == "test" ? filenames[2] : filenames[0];
  }

  const std::string& labelFile(std::string_view type) const {
    return type == "test" ? filenames[3] : filenames[1];
  }

  /// Drops the trailing ".gz" extension from a filename.
  static std::string stripGz(std::string_view name) {
    std::string_view ext = ".gz";
    if (name.size() >= ext.size() &&
        name.substr(name.size() - ext.size()) == ext) {
      name.remove_suffix(ext.size());
    }
    return std::string(name);
  }

  /// Reads a big-endian 32-bit integer (the format MNIST headers use).
  static std::int32_t readBigEndian32(std::ifstream& in) {
    std::int32_t value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
      throw std::runtime_error("Unexpected end of file while reading header");
    }
    return std::byteswap(value);
  }

  std::ifstream openBinary(const std::string& filename) const {
    std::string path = datasetDir + "/" + stripGz(filename);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      throw std::runtime_error("Failed to open file: " + path);
    }
    return in;
  }

  /// Runs a shell command, throwing `errorMessage` on non-zero exit.
  static void run(const std::string& command, const std::string& errorMessage) {
    if (std::system(command.c_str()) != 0) {
      throw std::runtime_error(errorMessage);
    }
  }

  /// Downloads and unzips a single archive if it is not already present.
  void download(const std::string& filename) const {
    std::string target = datasetDir + "/" + stripGz(filename);
    if (std::filesystem::exists(target)) {
      std::cout << target << " already downloaded\n";
      return;
    }

    std::string archive = datasetDir + "/" + filename;
    run("curl -L -o " + archive + " \"" + url + "/" + filename + "\"",
        "Failed to download " + filename);
    run("gzip -d " + archive, "Failed to unzip " + filename);
  }

  /// Reads an MNIST image file into a (nImages, 1, H+2*pad, W+2*pad) tensor,
  /// capped at the number of images this dataset was sized for.
  Tensor loadImages(const std::string& filename, int pad) {
    std::ifstream in = openBinary(filename);

    readBigEndian32(in);  // magic number
    int nImages = std::min(features.getDimensions()[0], readBigEndian32(in));
    int nRows = readBigEndian32(in);
    int nCols = readBigEndian32(in);

    Tensor images({nImages, 1, nRows + 2 * pad, nCols + 2 * pad});
    std::vector<unsigned char> buffer(static_cast<size_t>(nRows) * nCols);

    for (int i = 0; i < nImages; i++) {
      in.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
      for (int r = 0; r < nRows; r++) {
        for (int c = 0; c < nCols; c++) {
          images[{i, 0, pad + r, pad + c}] = buffer[r * nCols + c] / 255.0;
        }
      }
    }
    return images;
  }

  /// Reads an MNIST label file into a one-hot (nLabels, numClasses) tensor.
  Tensor loadLabels(const std::string& filename) {
    std::ifstream in = openBinary(filename);

    readBigEndian32(in);  // magic number
    int nLabels = std::min(labels.getDimensions()[0], readBigEndian32(in));

    Tensor oneHot({nLabels, numClasses});
    for (int i = 0; i < nLabels; i++) {
      unsigned char label;
      in.read(reinterpret_cast<char*>(&label), 1);
      oneHot[{i, label}] = 1.0;
    }
    return oneHot;
  }

 public:
  MNISTFashionDataset(int nImages, int batchSize, int imageHeight,
                      int imageWidth)
      : dataset(batchSize, Tensor({nImages, imageHeight, imageWidth}),
                Tensor({nImages, 1})) {}

  void downloadFashionMNIST(const std::string& directory) {
    datasetDir = directory;
    std::filesystem::create_directories(datasetDir);
    for (const auto& filename : filenames) {
      download(filename);
    }
  }

  void loadFashionMNIST(std::string type = "train", int pad = 0) {
    features = loadImages(imageFile(type), pad);
    labels = loadLabels(labelFile(type));
  }

  std::string textLabel(int i) { return std::string(classNames[i]); }

  std::string textLabel(Tensor oneHot) {
    auto begin = oneHot.getData();
    auto ptr = std::find(begin, begin + oneHot.getNValues(), 1.0);
    return std::string(classNames[ptr - begin]);
  }

  void plotImage(int index) {
    Tensor image = features[index][0];

    auto heat =
        plotlypp::Heatmap().z(image).colorscale("Greys").reversescale(true);

    auto layout =
        plotlypp::Layout()
            .title(plotlypp::Layout::Title().text(
                "FashionMNIST image " + std::to_string(index) + " (" +
                textLabel(labels[index]) + ")"))
            .yaxis(plotlypp::Layout::Yaxis()
                       .autorange(plotlypp::Layout::Yaxis::Autorange::Reversed)
                       .scaleanchor("x"));

    auto figure = plotlypp::Figure()
                      .addTraces(std::vector<plotlypp::Trace>{std::move(heat)})
                      .setLayout(std::move(layout));

    figure.show();
  }
};
}  // namespace mattTorch
