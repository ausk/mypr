#ifndef PLATE_RECOGNISE_HPP
#define PLATE_RECOGNISE_HPP
#include "chars_recognise.hpp"
#include "plate_detect.hpp"

/*! \namespace easypr
   Namespace where all the C++ EasyPR functionality resides
 */
namespace mypr {

class CPlateRecognise : public CPlateDetect, public CCharsRecognise {
public:
  CPlateRecognise();

  int plateRecognize(Mat src, std::vector<CPlate> &plateVec, int img_index = 0);
  int plateRecognize(Mat src, std::vector<std::string> &licenseVec);
  int plateRecognize(Mat src, std::vector<CPlate> &plateVec, int img_index,
                     Mat &output, const vector<CPlate> &plateVecGT);

  int plateRecognizeAsText(Mat src, std::vector<CPlate> &licenseVec);
  int plateRecognizeAsTextNM(Mat src, std::vector<CPlate> &licenseVec);

  inline void setLifemode(bool param) { CPlateDetect::setPDLifemode(param); }
  inline void setDetectType(int param) { CPlateDetect::setDetectType(param); }

  inline void setResultShow(bool param) { m_showResult = param; }
  inline bool getResultShow() const { return m_showResult; }

  inline void setDetectShow(bool param) { CPlateDetect::setDetectShow(param); }
  inline void setDebug(bool param) { setResultShow(param); }

  void LoadSVM(std::string path);
  void LoadANN(std::string path);
  void LoadChineseANN(std::string path);

private:
  // show the detect and recognition result image
  bool m_showResult;
};

} /* \namespace easypr  */

#endif // PLATE_RECOGNISE_HPP
