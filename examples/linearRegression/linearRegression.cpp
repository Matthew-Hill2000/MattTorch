#include <mattTorch/mattTorch.h>

#include <iostream>
#include <plotlypp/figure.hpp>
#include <plotlypp/traces/scatter.hpp>

#include "mattTorch/optimiser/sgd/sgd.h"
#include "syntheticRegressionDataset.h"

// Regression problems are concerned with the prediction of numerical values,
// though not every prediction problem is a regression problem. We typically
// might want to develop a model for predicting some numerical value, given
// some collection of input values. When training the model to perform this
// prediction, the entire dataset is called the training dataset, each input to
// the model is known as an example. The expected output is known as the
// label and the associated input variables are known as the features.
//
// Linear regression is the simplest tool for tackling regession problems and it
// relies on a few simplifying assumptions. First, we assume that the
// relationship between the features and targets is linear, i.e that the
// conditional mean E[Y | X=x] can be expressed as a weighted sum of features.
// This allows that the target value may still deviate from its expected value
// on account of observation noise. Next, we assume that any such noies is well
// behaved, following a Gaussian distribution. We typically denote number of
// examples in our dataset with n and we use superscripts to enumerate samples
// and targets and subscripts to index coordinates. So, x^i_j denotes the jth
// coordinate of the ith sample.

int main() {
  double LEARNINGRATE{0.001};
  double WEIGHTDECAY{0.0};
  int NUMEPOCHS{100};
  int NUMSAMPLES{100};
  int BATCHSIZE{4};
  int NUMTRAINBATCHES = (NUMSAMPLES * 0.8 + (BATCHSIZE - 1)) / BATCHSIZE;
  int NUMVALBATCHES = (NUMSAMPLES * 0.2 + (BATCHSIZE - 1)) / BATCHSIZE;
  int NUMINPUTS{20};

  mattTorch::Network net =
      mattTorch::NetworkBuilder().addFullyConnectedLayer(NUMINPUTS, 1).build();

  mattTorch::SGD sgd(net.getParameters(), LEARNINGRATE, WEIGHTDECAY);
  mattTorch::criterion::MSELoss mse;

  // Generate training data as vectors of individual tensors

  std::vector<double> epochs;
  std::vector<double> trainLossValues;
  std::vector<double> valLossValues;

  mattTorch::Tensor weights({NUMINPUTS, 1});
  mattTorch::Tensor bias({1});

  weights[{0, 0}] = 2.0;
  weights[{1, 0}] = -3.4;
  weights[{2, 0}] = 2.1;
  weights[{3, 0}] = 8.2;
  weights[{4, 0}] = 1.1;
  weights[{5, 0}] = -2.3;
  weights[{6, 0}] = -3.9;
  weights[{7, 0}] = -2.4;
  weights[{8, 0}] = 2.4;
  weights[{9, 0}] = 12.4;
  weights[{10, 0}] = 13.4;
  weights[{11, 0}] = 3.0;
  weights[{12, 0}] = -1.4;
  weights[{13, 0}] = -2.4;
  weights[{14, 0}] = -3.4;
  weights[{15, 0}] = -4.4;
  weights[{16, 0}] = -5.4;
  weights[{17, 0}] = -6.4;
  weights[{18, 0}] = -7.4;
  weights[{19, 0}] = -8.4;

  bias.setValueDirect(0, 4.2);

  mattTorch::SyntheticRegressionDataset data(weights, bias, 0.01, 1000,
                                             BATCHSIZE);

  data.testValSplit(0.05);

  // Training loop
  for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
    double epoch_loss = 0.0;
    int epoch_count{0};

    for (int i = 0; i < NUMTRAINBATCHES; i++) {
      sgd.zeroGrad();

      std::vector<mattTorch::Tensor> batchData = data.getBatch();
      mattTorch::Tensor output = net.forward(batchData[0]);
      mattTorch::Tensor loss = mse.calculateLoss(output, batchData[1]);
      epoch_loss += loss.getData()[0];

      int n = batchData[0].getDimensions()[0];
      epoch_loss += loss.getData()[0] * n;
      epoch_count += n;

      loss.backward();

      sgd.updateParameters();
    }

    data.shuffle();

    trainLossValues.push_back(epoch_loss / epoch_count);
    epochs.push_back(epoch);

    double val_loss{0};
    int val_count{0};
    for (int i = 0; i < NUMVALBATCHES; i++) {
      std::vector<mattTorch::Tensor> batchData = data.getBatch(false);
      mattTorch::Tensor output = net.forward(batchData[0]);
      mattTorch::Tensor loss = mse.calculateLoss(output, batchData[1]);

      int n = batchData[0].getDimensions()[0];
      val_loss += loss.getData()[0] * n;
      val_count += n;
    }

    valLossValues.push_back(val_loss / val_count);

    if (epoch % 5 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << epoch_loss / epoch_count
                << " Validation Loss: " << val_loss / val_count << std::endl;
    }
  }

  // Test
  std::cout << "\nTesting:\n";

  std::cout << "Trained weight values: \n" << *(net.getParameters()[0]) << "\n";
  std::cout << "Trained bias values: \n" << *(net.getParameters()[1]) << "\n";

  // Plot
  auto trainPlot = plotlypp::Scatter()
                       .x(epochs)
                       .y(trainLossValues)
                       .mode({plotlypp::Scatter::Mode::Lines,
                              plotlypp::Scatter::Mode::Markers})
                       .name("Training Loss against Epoch");

  auto valPlot = plotlypp::Scatter()
                     .x(epochs)
                     .y(valLossValues)
                     .mode({plotlypp::Scatter::Mode::Lines,
                            plotlypp::Scatter::Mode::Markers})
                     .name("Validation Loss against Epoch");

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

  return 0;
}
