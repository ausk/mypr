#ifndef CHARACTER_HPP
#define CHARACTER_HPP
#include <opencv2/opencv.hpp>

namespace mypr {
class CCharacter {
private:
  cv::Mat m_characterMat;
  cv::Rect m_characterPos;
  cv::String m_characterStr;
  double m_score;
  bool m_isChinese;
  double m_ostuLevel;
  cv::Point m_center;

public:
  CCharacter() {
    m_characterMat = cv::Mat();
    m_characterPos = cv::Rect();
    m_characterStr = "";
    m_score = 0;
    m_isChinese = false;
    m_ostuLevel = 125;
    m_center = cv::Point(0, 0);
  }

  CCharacter(const CCharacter &other) {
    m_characterMat = other.m_characterMat;
    m_characterPos = other.m_characterPos;
    m_characterStr = other.m_characterStr;
    m_score = other.m_score;
    m_isChinese = other.m_isChinese;
    m_ostuLevel = other.m_ostuLevel;
    m_center = other.m_center;
  }

  inline void setCharacterMat(const cv::Mat &param) {
    m_characterMat = param.clone();
  }
  inline cv::Mat getCharacterMat() const { return m_characterMat; }

  inline void setCharacterPos(cv::Rect param) { m_characterPos = param; }
  inline cv::Rect getCharacterPos() const { return m_characterPos; }

  inline void setCharacterStr(cv::String param) { m_characterStr = param; }
  inline cv::String getCharacterStr() const { return m_characterStr; }

  inline void setCharacterScore(double param) { m_score = param; }
  inline double getCharacterScore() const { return m_score; }

  inline void setIsChinese(bool param) { m_isChinese = param; }
  inline bool getIsChinese() const { return m_isChinese; }

  inline void setOstuLevel(double param) { m_ostuLevel = param; }
  inline double getOstuLevel() const { return m_ostuLevel; }

  inline void setCenterPoint(cv::Point param) { m_center = param; }
  inline cv::Point getCenterPoint() const { return m_center; }

  inline bool getIsStrong() const { return m_score >= 0.9; }
  inline bool getIsWeak() const { return m_score < 0.9 && m_score >= 0.5; }
  inline bool getIsLittle() const { return m_score < 0.5; }

  bool operator<(const CCharacter &other) const {
    return m_score < other.m_score;
  }

  bool operator>(const CCharacter &other) const {
    return m_score > other.m_score;
  }
};
}

#endif // CHARACTER_HPP
