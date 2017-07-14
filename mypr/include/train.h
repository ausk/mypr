#ifndef TRAIN_HPP
#define TRAIN_HPP
#include <opencv2/opencv.hpp>
namespace mypr {
class VTrain {
public:
  VTrain(){};
  virtual ~VTrain(){};
  virtual void train() = 0;
  virtual void test() = 0;

private:
  virtual cv::Ptr<cv::ml::TrainData> tdata() = 0;
};
}

#endif // TRAIN_HPP
