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
  double LearningRate{0.01};
  int NUMEPOCHS{600};
  int NUMSAMPLES{1000};
  int BATCHSIZE{32};
  int NUMBATCHES = (NUMSAMPLES + (BATCHSIZE - 1)) / BATCHSIZE;
  int NUMINPUTS{2};

  mattTorch::Network net =
      mattTorch::NetworkBuilder().addFullyConnectedLayer(NUMINPUTS, 1).build();

  mattTorch::SGD sgd(net.getParameters(), LearningRate);
  mattTorch::criterion::MSELoss mse;

  // Generate training data as vectors of individual tensors

  std::vector<double> epochs;
  std::vector<double> lossValues;

  mattTorch::Tensor weights({NUMINPUTS, 1});
  mattTorch::Tensor bias({1});

  weights[{0, 0}] = 2.0;
  weights[{1, 0}] = -3.4;

  bias.setValueDirect(0, 4.2);

  mattTorch::SyntheticRegressionDataset data(weights, bias, 0.01, 1000,
                                             BATCHSIZE);

  // Training loop
  for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
    double epoch_loss = 0.0;

    for (int i = 0; i < NUMBATCHES; i++) {
      sgd.zeroGrad();

      std::vector<mattTorch::Tensor> batchData = data.getBatch();
      mattTorch::Tensor output = net.forward(batchData[0]);
      mattTorch::Tensor loss = mse.calculateLoss(output, batchData[1]);
      epoch_loss += loss.getData()[0];

      loss.backward();

      sgd.updateParameters();
    }
    data.shuffle();

    lossValues.push_back(epoch_loss / NUMBATCHES);
    epochs.push_back(epoch);

    if (epoch % 50 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << epoch_loss / NUMBATCHES
                << std::endl;
    }
  }

  // Test
  std::cout << "\nTesting:\n";

  std::cout << "Trained weight values: \n" << *(net.getParameters()[0]) << "\n";
  std::cout << "Trained bias values: \n" << *(net.getParameters()[1]) << "\n";

  // Plot
  auto scatterPlot = plotlypp::Scatter()
                         .x(epochs)
                         .y(lossValues)
                         .mode({plotlypp::Scatter::Mode::Lines,
                                plotlypp::Scatter::Mode::Markers})
                         .name("Loss against Epoch");

  auto layout = plotlypp::Layout()
                    .title(plotlypp::Layout::Title().text("Loss against epoch"))
                    .xaxis(plotlypp::Layout::Xaxis().title(
                        plotlypp::Layout::Xaxis::Title().text("Epoch")))
                    .yaxis(plotlypp::Layout::Yaxis().title(
                        plotlypp::Layout::Yaxis::Title().text("Loss")));

  auto figure = plotlypp::Figure()
                    .addTrace(std::move(scatterPlot))
                    .setLayout(std::move(layout));

  figure.show();

  figure.writeHtml("loss_against_epoch.html");

  return 0;
}
