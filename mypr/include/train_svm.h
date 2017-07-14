#ifndef TRAIN_SVM_HPP
#define TRAIN_SVM_HPP

#include "config.hpp"
#include "train.hpp"
#include <opencv2/opencv.hpp>

namespace mypr{
class CTrainSvm : public VTrain {
public:
  typedef struct {
    std::string filename;
    SvmLabel label;
  } TrainItem;

  CTrainSvm(const char *plates_folder, const char *svm_xml);
  virtual void train();
  virtual void test();

private:
  void prepare();
  virtual cv::Ptr<cv::ml::TrainData> tdata();
  cv::Ptr<cv::ml::SVM> m_svm;
  const std::string m_plates_folder;
  const std::string m_svm_xml;
  std::vector<TrainItem> m_train_file_list;
  std::vector<TrainItem> m_test_file_list;
};
}

#endif // TRAIN_SVM_HPP
