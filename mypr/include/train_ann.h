#ifndef TRAIN_ANN_HPP
#define TRAIN_ANN_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "train.hpp"
#include "config.hpp"
#include <memory>

namespace mypr {

class CTrainAnn : public VTrain {
public:
  explicit CTrainAnn(const char *chars_folder, const char *ann_xml);

  virtual void train();

  virtual void test();

  std::pair<std::string, std::string> identifyChinese(cv::Mat input);
  std::pair<std::string, std::string> identify(cv::Mat input);

private:
  virtual cv::Ptr<cv::ml::TrainData> tdata();

  cv::Ptr<cv::ml::TrainData> sdata(size_t number_for_count = 100);

  cv::Ptr<cv::ml::ANN_MLP> m_ann;
  const std::string m_ann_xml;
  const std::string m_chars_folder;

  int type;
};
}

#endif // TRAIN_ANN_HPP
