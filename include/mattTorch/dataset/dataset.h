#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch {
class dataset {
 private:
  TensorView examples;
  TensorView labels;
  int numExamples;
  int exampleSize;
  std::vector<int> indices;
  int batchSize;

 public:
  dataset(int batchSize);
  std::vector<TensorView> getBatch(int batchIndex);
  void shuffle();
  void loadData(const std::string& csvPath);
  void printNumber();
};
}  // namespace mattTorch
