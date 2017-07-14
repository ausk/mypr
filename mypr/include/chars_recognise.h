#ifndef CHARS_RECOGNISE_HPP
#define CHARS_RECOGNISE_HPP
#include "chars_identify.hpp"
#include "chars_segment.hpp"
#include "config.hpp"
#include "core_func.hpp"
#include "plate.hpp"
#include "util.hpp"
#include <map>
#include <memory>

namespace mypr {

class CCharsRecognise {
public:
  CCharsRecognise();

  ~CCharsRecognise();

  int charsRecognise(cv::Mat plate, std::string &plateLicense);
  int charsRecognise(CPlate &plate, std::string &plateLicense);

  static std::map<int, std::string> colors;

  inline std::string getPlateColor(cv::Mat input) const {
    return colors[getPlateType(input, true)];
  }

  inline std::string getPlateColor(Color in) const { return colors[in]; }

  inline void setLiuDingSize(int param) {
    m_charsSegment->setLiuDingSize(param);
  }
  inline void setColorThreshold(int param) {
    m_charsSegment->setColorThreshold(param);
  }
  inline void setBluePercent(float param) {
    m_charsSegment->setBluePercent(param);
  }
  inline float getBluePercent() const {
    return m_charsSegment->getBluePercent();
  }
  inline void setWhitePercent(float param) {
    m_charsSegment->setWhitePercent(param);
  }
  inline float getWhitePercent() const {
    return m_charsSegment->getWhitePercent();
  }

private:
  //！字符分割

  std::shared_ptr<CCharsSegment> m_charsSegment;
   //CCharsSegment *m_charsSegment = new CCharsSegment();
   //SAFE_RELEASE(m_charsSegment);

};

} /* \namespace mypr  */
#endif // CHARS_RECOGNISE_HPP
