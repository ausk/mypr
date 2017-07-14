#ifndef CHARS_IDENTIFY_HPP
#define CHARS_IDENTIFY_HPP
#include "character.hpp"
#include "config.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
namespace mypr {

class CharsIdentify {
private:
  CharsIdentify();

  static std::shared_ptr<CharsIdentify> m_instance;
  cv::Ptr<cv::ml::ANN_MLP> m_ann;
  cv::Ptr<cv::ml::ANN_MLP> m_annChinese;

public:
  int classify(cv::Mat f, float &maxVal, bool isChinses = false);
  void classify(cv::Mat featureRows, std::vector<int> &out_maxIndexs,
                std::vector<float> &out_maxVals,
                std::vector<bool> isChineseVec);
  void classify(std::vector<CCharacter> &charVec);
  void classifyChinese(std::vector<CCharacter> &charVec);

  std::pair<std::string, std::string> identify(cv::Mat input,
                                               bool isChinese = false);
  int identify(std::vector<cv::Mat> inputs,
               std::vector<std::pair<std::string, std::string>> &outputs,
               std::vector<bool> isChineseVec);

  std::pair<std::string, std::string>
  identifyChinese(cv::Mat input, float &result, bool &isChinese);

  bool isCharacter(cv::Mat input, std::string &label, float &maxVal,
                   bool isChinese = false);

  void LoadModel(std::string path);
  void LoadChineseModel(std::string path);

  static std::shared_ptr<CharsIdentify> instance();
};
}
#endif // CHARS_IDENTIFY_HPP
