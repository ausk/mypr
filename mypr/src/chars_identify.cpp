#include "chars_identify.hpp"
#include "character.hpp"
#include "config.hpp"
#include "params.hpp"

#include "core_func.hpp"
#include "feature.hpp"

#include <opencv2/opencv.hpp>
namespace mypr {

std::shared_ptr<CharsIdentify> CharsIdentify::m_instance = nullptr;

std::shared_ptr<CharsIdentify> CharsIdentify::instance() {
  if (!m_instance.get()) {
    m_instance = std::make_shared<CharsIdentify>(CharsIdentify());
  }
  return m_instance;
}

CharsIdentify::CharsIdentify() {
  m_ann = cv::Algorithm::load<cv::ml::ANN_MLP>(kDefaultAnnPath);
  m_annChinese = cv::Algorithm::load<cv::ml::ANN_MLP>(kChineseAnnPath);
}

void CharsIdentify::LoadModel(std::string path) {
  if (path != std::string(kDefaultAnnPath)) {
    if (!m_ann->empty()) {
      m_ann->clear();
    }
    m_ann = cv::Algorithm::load<cv::ml::ANN_MLP>(path);
  }
}

void CharsIdentify::LoadChineseModel(std::string path) {
  if (path != std::string(kChineseAnnPath)) {
    if (!m_annChinese->empty()) {
      m_annChinese->clear();
    }
    m_annChinese = cv::Algorithm::load<cv::ml::ANN_MLP>(path);
  }
}

void CharsIdentify::classify(cv::Mat featureRows,
                             std::vector<int> &out_maxIndexs,
                             std::vector<float> &out_maxVals,
                             std::vector<bool> isChineseVec) {
  int rowNum = featureRows.rows;

  cv::Mat output(rowNum, kCharsTotalNumber, CV_32FC1);
  m_ann->predict(featureRows, output);


  for (int index = 0; index < rowNum; index++) {
    Mat output_row = output.row(index);
    float maxVal = -2.f;
    bool isChinses = isChineseVec[index];
    int start = 0, end = 0, maxIndex = -1;
    if (!isChinses) {
      start = 0;
      end = kCharactersNumber;
    } else {
      start = kCharactersNumber;
      end = kCharsTotalNumber;
    }

    for (int j = start, maxIndex = start; j < end; ++j) {
      float val = output_row.at<float>(j);
      if (val > maxVal) {
        maxVal = val;
        maxIndex = j;
      }
    }
    out_maxIndexs[index] = maxIndex;
    out_maxVals[index] = maxVal;
  }
}

void CharsIdentify::classify(std::vector<CCharacter> &charVec) {
  size_t charVecSize = charVec.size();

  if (charVecSize == 0) {
    return;
  }

  Mat featureRows;
  for (size_t index = 0; index < charVecSize; index++) {
    Mat charInput = charVec[index].getCharacterMat();
    Mat feature = charFeatures(charInput, kPredictSize);
    featureRows.push_back(feature);
  }

  cv::Mat output(charVecSize, kCharsTotalNumber, CV_32FC1);
  m_ann->predict(featureRows, output);

  for (size_t output_index = 0; output_index < charVecSize; output_index++) {
    CCharacter &character = charVec[output_index];
    Mat output_row = output.row(output_index);

    int start, end, maxIndex;
    float maxVal = -2.f;
    std::string label = "";

    bool isChinses = character.getIsChinese();
    if (!isChinses) {
      start = 0;
      end = kCharactersNumber;
    } else {
      start = kCharactersNumber;
      end = kCharsTotalNumber;
    }

    for (int j = start, maxIndex = start; j < end; ++j) {
      float val = output_row.at<float>(j);
      if (val > maxVal) {
        maxVal = val;
        maxIndex = j;
      }
    }
    label = CHARS_DICT.at(kChars[maxIndex]);

    character.setCharacterScore(maxVal);
    character.setCharacterStr(label);
  }
}

void CharsIdentify::classifyChinese(std::vector<CCharacter> &charVec) {
  size_t charVecSize = charVec.size();

  if (charVecSize == 0) {
    return;
  }

  Mat featureRows;
  for (size_t index = 0; index < charVecSize; index++) {
    Mat charInput = charVec[index].getCharacterMat();
    Mat feature = charFeatures(charInput, kChineseSize);
    featureRows.push_back(feature);
  }

  cv::Mat output(charVecSize, kChineseNumber, CV_32FC1);
  m_annChinese->predict(featureRows, output);

  for (size_t output_index = 0; output_index < charVecSize; output_index++) {
    CCharacter &character = charVec[output_index];
    Mat output_row = output.row(output_index);

    bool isChinese = true;
    float maxVal = -2;
    int result = -1;

    for (int j = 0; j < kChineseNumber; j++) {
      float val = output_row.at<float>(j);
      if (val > maxVal) {
        maxVal = val;
        result = j;
      }
    }

    // no match
    if (-1 == result) {
      result = 0;
      maxVal = 0;
      isChinese = false;
    }

    auto index = result + kCharactersNumber;

    std::string province = CHARS_DICT.at(kChars[index]);
    character.setCharacterScore(maxVal);
    character.setCharacterStr(province);
    character.setIsChinese(true);
  }
}

int CharsIdentify::classify(cv::Mat f, float &maxVal, bool isChinses) {
  cv::Mat output(1, kCharsTotalNumber, CV_32FC1);
  m_ann->predict(f, output);

  maxVal = -2.f;
  int start = 0, end = 0, maxIndex = -1;
  if (!isChinses) {
    start = 0;
    end = kCharactersNumber;
  } else {
    start = kCharactersNumber;
    end = kCharsTotalNumber;
  }

    for (int j = start; j < end; ++j) {
      float val = output.at<float>(j);
      if (val > maxVal) {
        maxVal = val;
        maxIndex = j;
      }
    }

  return maxIndex;
}

bool CharsIdentify::isCharacter(cv::Mat input, std::string &label,
                                float &maxVal, bool isChinese) {
  cv::Mat feature = charFeatures(input, kPredictSize);
  auto index = static_cast<int>(classify(feature, maxVal, isChinese));

  float chineseMaxThresh = 0.2f;
  // float chineseMaxThresh = CParams::instance()->getParam2f();

  if (maxVal >= 0.9 || (isChinese && maxVal >= chineseMaxThresh)) {

    label = CHARS_DICT.at(kChars[index]);
    return true;
  } else {
    return false;
  }
}

std::pair<std::string, std::string>
CharsIdentify::identifyChinese(cv::Mat input, float &out, bool &isChinese) {
  cv::Mat feature = charFeatures(input, kChineseSize);
  float maxVal = -2;

  int result = -1;

  cv::Mat output(1, kChineseNumber, CV_32FC1);
  m_annChinese->predict(feature, output);

  // 2017.01.07 gailv
  for (int j = 0; j < kChineseNumber; j++) {
    float val = output.at<float>(j);
    if (val > maxVal) {
      maxVal = val;
      result = j;
    }
  }

  // no match
  if (-1 == result) {
    result = 0;
    maxVal = 0;
    isChinese = false;
  } else if (maxVal > 0.9) {
    isChinese = true;
  }

  out = maxVal;
  auto index = result + kCharactersNumber;
  std::string key = kChars[index];
  std::string province = CHARS_DICT.at(key);
  return std::make_pair(key, province);
}

std::pair<std::string, std::string> CharsIdentify::identify(cv::Mat input,
                                                            bool isChinese) {
  cv::Mat feature = charFeatures(input, kPredictSize);
  float maxVal = -2;
  auto index = static_cast<int>(classify(feature, maxVal, isChinese));
  std::string key = kChars[index];
  std::string province = CHARS_DICT.at(key);
  return std::make_pair(key, province);
}

int CharsIdentify::identify(
    std::vector<cv::Mat> inputs,
    std::vector<std::pair<std::string, std::string>> &outputs,
    std::vector<bool> isChineseVec) {
  Mat featureRows;
  size_t input_size = inputs.size();
  for (size_t i = 0; i < input_size; i++) {
    Mat input = inputs[i];
    cv::Mat feature = charFeatures(input, kPredictSize);
    featureRows.push_back(feature);
  }

  std::vector<int> maxIndexs;
  std::vector<float> maxVals;
  classify(featureRows, maxIndexs, maxVals, isChineseVec);

  for (size_t row_index = 0; row_index < input_size; row_index++) {
    int index = maxIndexs[row_index];
    std::string key = kChars[index];
    std::string value = CHARS_DICT.at(key);
    outputs[row_index] = std::make_pair(key, value);
  }
  return 0;
}
}
