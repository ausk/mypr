#include "chars_recognise.hpp"
#include "character.hpp"
#include "util.hpp"

namespace mypr {

std::map<int, std::string> CCharsRecognise::colors = {{mypr::BLUE, "蓝牌"},
                                                      {mypr::YELLOW, "黄牌"},
                                                      {mypr::WHITE, "白牌"},
                                                      {mypr::UNKNOWN, "未知"}};

CCharsRecognise::CCharsRecognise() {
  m_charsSegment = std::make_shared<CCharsSegment>(CCharsSegment());
}

CCharsRecognise::~CCharsRecognise() {}

int CCharsRecognise::charsRecognise(Mat plate, std::string &plateLicense) {
  std::vector<Mat> matChars;

  int result = m_charsSegment->charsSegment(plate, matChars);

  if (result == 0) {
    int num = matChars.size();
    for (int j = 0; j < num; j++) {
      Mat charMat = matChars.at(j);
      bool isChinses = false;
      float maxVal = 0;
      if (j == 0) {
        isChinses = true;
        auto character =
            CharsIdentify::instance()->identifyChinese(charMat, maxVal, isChinses);
        plateLicense.append(character.second);
      } else {
        isChinses = false;
        auto character =
            CharsIdentify::instance()->identify(charMat, isChinses);
        plateLicense.append(character.second);
      }
    }
  }
  if (plateLicense.size() < 7) {
    return -1;
  }

  return result;
}

int CCharsRecognise::charsRecognise(CPlate &plate, std::string &plateLicense) {
  std::vector<Mat> matChars;

  Mat plateMat = plate.getPlateMat();

  Color color;
  if (plate.getPlateLocateType() == CMSER) {
    color = plate.getPlateColor();
  } else {
    int w = plateMat.cols;
    int h = plateMat.rows;
    Mat tmpMat = plateMat(Rect_<double>(w * 0.1, h * 0.1, w * 0.8, h * 0.8));
    color = getPlateType(tmpMat, true);
  }

  int result = m_charsSegment->charsSegment(plateMat, matChars, color);
  if (result == 0) {
    int num = matChars.size();
    for (int j = 0; j < num; j++) {
      Mat charMat = matChars.at(j);
      bool isChinses = false;
      std::pair<std::string, std::string> character;
      float maxVal;
      if (j == 0) {
        isChinses = true;
        character =
            CharsIdentify::instance()->identifyChinese(charMat, maxVal, isChinses);
        plateLicense.append(character.second);
      } else {
        isChinses = false;
        character = CharsIdentify::instance()->identify(charMat, isChinses);
        plateLicense.append(character.second);
      }

      CCharacter charResult;
      charResult.setCharacterMat(charMat);
      charResult.setCharacterStr(character.second);

      plate.addReutCharacter(charResult);
    }
    if (plateLicense.size() < 7) {
      return -1;
    }
  }

  return result;
}
}
