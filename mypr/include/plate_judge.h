#ifndef PLATE_JUDGE_HPP
#define PLATE_JUDGE_HPP
#include "plate.hpp"
#include "feature.hpp"
#include <memory>

namespace mypr {

class CPlateJudge {
public:
  static std::shared_ptr<CPlateJudge> instance();

  void LoadModel(std::string svmxml);

  int plateJudge(const std::vector<CPlate> &, std::vector<CPlate> &);
  int plateJudgeUsingNMS(const std::vector<CPlate> &, std::vector<CPlate> &,
                         int maxPlates = 5);

  int plateJudge(const std::vector<Mat> &, std::vector<Mat> &);

  int plateJudge(const Mat &inMat, int &result);
  int plateSetScore(CPlate &plate);

private:
  CPlateJudge();

  static std::shared_ptr<CPlateJudge> m_instance;

  svmCallback extractFeature;

  cv::Ptr<ml::SVM> m_svm;
};
}
#endif // PLATE_JUDGE_HPP
