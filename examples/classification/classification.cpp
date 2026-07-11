#include <mattTorch/mattTorch.h>

#include <iostream>
#include <plotlypp/figure.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <vector>

#include "mattTorch/criterion/crossEntropyLoss/softmaxCrossEntropyLoss.h"
#include "mnistFashionDataset.h"

namespace {

using mattTorch::Tensor;

// ---- Hyperparameters ----
constexpr double LEARNING_RATE = 0.01;
constexpr double WEIGHT_DECAY = 0.0;
constexpr int NUM_EPOCHS = 50;
constexpr double TRAIN_VAL_SPLIT = 0.8;

// ---- Data geometry ----
constexpr int NUM_TRAIN = 4000;
constexpr int NUM_TEST = 10000;
constexpr int IMAGE_ROWS = 28;
constexpr int IMAGE_COLS = 28;
constexpr int PADDING = 0;
constexpr int BATCH_SIZE = 256;
constexpr int NUM_CLASSES = 10;
constexpr int INPUT_SIZE =
    (IMAGE_ROWS + 2 * PADDING) * (IMAGE_COLS + 2 * PADDING);

int numBatches(int numExamples) {
  return (numExamples + BATCH_SIZE - 1) / BATCH_SIZE;
}

Tensor flatten(Tensor& batch) {
  return batch.reshape({batch.getDimensions()[0], INPUT_SIZE});
}

void plotLosses(const std::vector<double>& epochs,
                const std::vector<double>& trainLoss,
                const std::vector<double>& valLoss) {
  auto trainPlot = plotlypp::Scatter()
                       .x(epochs)
                       .y(trainLoss)
                       .mode({plotlypp::Scatter::Mode::Lines,
                              plotlypp::Scatter::Mode::Markers})
                       .name("Training Loss");
  auto valPlot = plotlypp::Scatter()
                     .x(epochs)
                     .y(valLoss)
                     .mode({plotlypp::Scatter::Mode::Lines,
                            plotlypp::Scatter::Mode::Markers})
                     .name("Validation Loss");
  auto layout = plotlypp::Layout()
                    .title(plotlypp::Layout::Title().text("Loss against epoch"))
                    .xaxis(plotlypp::Layout::Xaxis().title(
                        plotlypp::Layout::Xaxis::Title().text("Epoch")))
                    .yaxis(plotlypp::Layout::Yaxis().title(
                        plotlypp::Layout::Yaxis::Title().text("Loss")));
  auto figure = plotlypp::Figure()
                    .addTraces(std::vector<plotlypp::Trace>{
                        std::move(trainPlot), std::move(valPlot)})
                    .setLayout(std::move(layout));
  figure.show();
  figure.writeHtml("loss_against_epoch.html");
}

}  // namespace

int main() {
  mattTorch::MNISTFashionDataset data(NUM_TRAIN, BATCH_SIZE, IMAGE_ROWS,
                                      IMAGE_COLS);
  data.downloadFashionMNIST("examples/classification/dataset");
  data.loadFashionMNIST("train", PADDING);
  data.testValSplit(TRAIN_VAL_SPLIT);

  mattTorch::MNISTFashionDataset testData(NUM_TEST, BATCH_SIZE, IMAGE_ROWS,
                                          IMAGE_COLS);
  testData.loadFashionMNIST("test", PADDING);

  mattTorch::Network net = mattTorch::NetworkBuilder()
                               .addFullyConnectedLayer(INPUT_SIZE, 128, "he")
                               .addReluLayer()
                               .addFullyConnectedLayer(128, NUM_CLASSES, "he")
                               .build();

  mattTorch::SGD sgd(net.getParameters(), LEARNING_RATE, WEIGHT_DECAY);
  mattTorch::criterion::SoftmaxCrossEntropyLoss lossFunc;

  // One pass over `batches` batches; trains when `train` is set. Returns the
  // mean loss per example.
  auto runEpoch = [&](mattTorch::MNISTFashionDataset& dataset, int batches,
                      bool train) {
    double totalLoss = 0.0;
    int totalCount = 0;
    for (int batch = 0; batch < batches; batch++) {
      if (train) sgd.zeroGrad();

      std::vector<Tensor> batchData = dataset.getBatch(train);
      Tensor features = flatten(batchData[0]);
      Tensor output = net.forward(features);
      Tensor loss = lossFunc.calculateLoss(output, batchData[1]);

      int n = batchData[0].getDimensions()[0];
      totalLoss += loss.getData()[0] * n;
      totalCount += n;

      if (train) {
        loss.backward();
        sgd.updateParameters();
      }
    }
    return totalLoss / totalCount;
  };

  const int trainExamples = static_cast<int>(NUM_TRAIN * TRAIN_VAL_SPLIT);
  const int valExamples = NUM_TRAIN - trainExamples;

  std::vector<double> epochs, trainLoss, valLoss;
  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    double trainEpochLoss = runEpoch(data, numBatches(trainExamples), true);
    data.shuffle();
    double valEpochLoss = runEpoch(data, numBatches(valExamples), false);

    epochs.push_back(epoch);
    trainLoss.push_back(trainEpochLoss);
    valLoss.push_back(valEpochLoss);

    if (epoch % 5 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << trainEpochLoss
                << " Validation Loss: " << valEpochLoss << std::endl;
    }
  }

  // ---- Test accuracy ----
  int correct = 0;
  for (int batch = 0; batch < numBatches(NUM_TEST); batch++) {
    std::vector<Tensor> batchData = testData.getBatch();
    Tensor features = flatten(batchData[0]);
    Tensor output = net.forward(features);
    Tensor& labels = batchData[1];

    for (int i = 0; i < output.getDimensions()[0]; i++) {
      if (output[i].argmax() == labels[i].argmax()) correct++;
    }
  }
  std::cout << "Accuracy: " << 100.0 * correct / NUM_TEST << "%" << std::endl;

  plotLosses(epochs, trainLoss, valLoss);
  return 0;
}
