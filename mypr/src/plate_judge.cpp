#include "plate_judge.hpp"
#include "config.hpp"
#include "core_func.hpp"
#include "params.hpp"
#include <opencv2/opencv.hpp>

namespace mypr {

std::shared_ptr<CPlateJudge> CPlateJudge::m_instance = nullptr;

std::shared_ptr<CPlateJudge> CPlateJudge::instance() {
  if (!m_instance) {
    m_instance = std::make_shared<CPlateJudge>(CPlateJudge());
  }
  return m_instance;
}

CPlateJudge::CPlateJudge() {
  m_svm = Algorithm::load<ml::SVM>(kDefaultSvmPath);
  // m_svm = ml::SVM::load<ml::SVM>(kLBPSvmPath);
  extractFeature = getLBPFeatures;
}

void CPlateJudge::LoadModel(std::string svmxml) {
  if (svmxml != std::string(kDefaultSvmPath)) {
    if (!m_svm->empty()) {
      m_svm->clear();
    }
    m_svm = Algorithm::load<ml::SVM>(svmxml);
  }
}

int CPlateJudge::plateJudge(const Mat &inMat, int &result) {
  //对单图像块judge
  Mat features;
  extractFeature(inMat, features);

  float response_score = m_svm->predict(features);
  result = (int)response_score;
  return 0;
}

int CPlateJudge::plateJudge(const std::vector<Mat> &inVec,
                            std::vector<Mat> &resultVec) {
  //对图像块序列judege
  int num = inVec.size();
  for (int j = 0; j < num; j++) {
    Mat inMat = inVec[j];

    int response = -1;
    plateJudge(inMat, response);

    if (response == 1) {
      resultVec.push_back(inMat);
    }
  }
  return 0;
}

// set the score of plate
// 0 is plate, -1 is not.
int CPlateJudge::plateSetScore(CPlate &plate) {
  Mat features;
  extractFeature(plate.getPlateMat(), features);

  float score =
      m_svm->predict(features, noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);

  // std::cout << "score:" << score << std::endl;

  // score is the distance of margin，below zero is plate, up is not
  // when score is below zero, the samll the value, the more possibliy to be a
  // plate.
  plate.setPlateScore(score);

  if (score < 0) {
    return 0;
  } else {
    return -1;
  }
}

// non-maximum suppression
void NMS(std::vector<CPlate> &inVec, std::vector<CPlate> &resultVec,
         double overlap_threshold) {
    resultVec.clear();

  std::sort(inVec.begin(), inVec.end());

  std::vector<CPlate>::iterator it = inVec.begin();
  for (; it != inVec.end(); ++it) {
    CPlate plateSrc = *it;
    // std::cout << "plateScore:" << plateSrc.getPlateScore() << std::endl;
    Rect rectSrc = plateSrc.getPlatePos().boundingRect();

    std::vector<CPlate>::iterator itc = it + 1;

    for (; itc != inVec.end();) {
      CPlate plateComp = *itc;
      Rect rectComp = plateComp.getPlatePos().boundingRect();
      // Rect rectInter = rectSrc & rectComp;
      // Rect rectUnion = rectSrc | rectComp;
      // double r = double(rectInter.area()) / double(rectUnion.area());
      float iou = computeIOU(rectSrc, rectComp);

      if (iou > overlap_threshold) {
        itc = inVec.erase(itc);
      } else {
        ++itc;
      }
    }
  }

  resultVec = inVec;
}

int CPlateJudge::plateJudgeUsingNMS(const std::vector<CPlate> &inVec,
                                    std::vector<CPlate> &resultVec,
                                    int maxPlates) {
  std::vector<CPlate> plateVec;
  int num = inVec.size();
  bool outputResult = false;

  bool useCascadeJudge = true;
  bool useShirkMat = true;

  for (int j = 0; j < num; j++) {
    CPlate plate = inVec[j];
    Mat inMat = plate.getPlateMat();

    int result = plateSetScore(plate);

    if (result == 0) {

#ifdef _DEBUG
      if (0) {
        imshow("inMat", inMat);
        waitKey(0);
        destroyWindow("inMat");
      }
#endif//_DEBUG

      if (plate.getPlateLocateType() == CMSER) {
        int w = inMat.cols;
        int h = inMat.rows;

        Mat tmpmat = inMat(Rect_<double>(w * 0.05, h * 0.1, w * 0.9, h * 0.8));
        Mat tmpDes = inMat.clone();
        resize(tmpmat, tmpDes, Size(inMat.size()));

        plate.setPlateMat(tmpDes);

        if (useCascadeJudge) {
          int resultCascade = plateSetScore(plate);

          if (plate.getPlateLocateType() != CMSER) {
            plate.setPlateMat(inMat);
          }

          if (resultCascade == 0) {
#ifdef _DEBUG
            {
              imshow("tmpDes", tmpDes);
              waitKey(0);
              destroyWindow("tmpDes");
            }
#endif //_DEBUG

            plateVec.push_back(plate);
          }
        } else {
          plateVec.push_back(plate);
        }
      } else {
        plateVec.push_back(plate);
      }
    }
  }

  std::vector<CPlate> dumpPlateVec;

  double overlap = 0.7;
  NMS(plateVec, dumpPlateVec, overlap);

  std::vector<CPlate>::iterator it = dumpPlateVec.begin();
  int count = 0;
  for (; it != dumpPlateVec.end() && count <= maxPlates; ++it,++count) {
    resultVec.push_back(*it);

#ifdef _DEBUG
    {
      imshow("plateMat", it->getPlateMat());
      waitKey(0);
      destroyWindow("plateMat");
    }
#endif //_DEBUG

  }

  return 0;
}

int CPlateJudge::plateJudge(const std::vector<CPlate> &inVec,
                            std::vector<CPlate> &resultVec) {
  int num = inVec.size();
  for (int j = 0; j < num; j++) {
    CPlate inPlate = inVec[j];
    Mat inMat = inPlate.getPlateMat();

    int response = -1;
    plateJudge(inMat, response);

    if (response == 1) {
      resultVec.push_back(inPlate);
    } else {
      int w = inMat.cols;
      int h = inMat.rows;

      Mat tmpmat = inMat(Rect_<double>(w * 0.05, h * 0.1, w * 0.9, h * 0.8));
      Mat tmpDes = inMat.clone();
      resize(tmpmat, tmpDes, Size(inMat.size()));

      plateJudge(tmpDes, response);

      if (response == 1) {
        resultVec.push_back(inPlate);
      }
    }
  }
  return 0;
}
}
