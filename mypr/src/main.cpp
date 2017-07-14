#include "plate_recognise.hpp"
#include "train_ann.hpp"
#include "train_svm.hpp"
#include "util.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Util;
using namespace mypr;
using namespace cv;
int test_train_model() {

  std::string svm_model =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/svm_model.xml";
  std::string ann_model =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model.xml";

  //  CTrainSvm svm("/home/auss/Projects/Qt/EasyPR/mypr/resources/train/svm",
  //                svm_model.c_str());
  //  svm.train();
  //  svm.test();

  CTrainAnn ann("/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann",
                ann_model.c_str());
  LOG(INFO) << "Start training ann ...\n";
  ann.train();
  LOG(INFO) << "Finished training ann ...\n";
  ann.test();
  return 0;
}

int test_ann() {

  std::string sub_folder = "/home/auss/Projects/Qt/EasyPR/ann_test/";
  std::string svm_model =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/svm_model.xml";
  std::string ann_model_10x10 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_10x10.xml";
  std::string ann_model_20x20 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_20x20.xml";

  CharsIdentify::instance()->LoadModel(ann_model_10x10);
  CPlateJudge::instance()->LoadModel(svm_model);

  cv::Ptr<cv::ml::ANN_MLP> m_ann;
  m_ann = cv::Algorithm::load<cv::ml::ANN_MLP>(ann_model_10x10);

  auto chars_files = Util::getFiles(sub_folder);
  for (auto filename : chars_files) {
    std::cout << filename << std::endl;
    Mat src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    cv::imshow("source", src);
    cv::waitKey(0);
    cv::Mat feature = charFeatures(src, kPredictSize);
    cv::Mat result(1, kCharsTotalNumber, CV_32FC1);
    m_ann->predict(feature, result);
    float prob = 0, maxProb = 0;
    int maxIndex = 0;
    for (int i = 0; i < kCharsTotalNumber; ++i) {
      prob = result.at<float>(i);
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    }
    std::string key = kChars[maxIndex];
    std::string value = CHARS_DICT.at(key);
    cout << key << " : " << value << std::endl;
  }
  LOG(INFO) << "Finished testing ann ...\n";
}

int test_pr() {
  std::string svm_model =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/svm_model.xml";
  std::string ann_model_10x10 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_10x10.xml";
  std::string ann_model_20x20 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_20x20.xml";

  CharsIdentify::instance()->LoadModel(ann_model_10x10);
  CPlateJudge::instance()->LoadModel(svm_model);

  LOG(INFO) << "Start test_pr()" << endl;
  const string plate_file_name = "resources/image/test_chars_SuEUK722.jpg";
  Mat src = imread(plate_file_name);
  imshow(plate_file_name, src);
  waitKey(0);

  CPlateLocate plate;
  std::vector<cv::Mat> resultVec1;
  int result;
  result = plate.plateLocate(src, resultVec1);

  if (result != 0) {
    LOG(ERROR) << "Cannot locate plates\n";
    return -1;
  }
  LOG(INFO) << "Locate plates\n";

  for (size_t i = 0, num = resultVec1.size(); i < num; ++i) {
    imshow(string("plate_locate_") + char('a' + i), resultVec1[i]);
    waitKey(0);
  }

  std::vector<cv::Mat> resultVec2;
  result = CPlateJudge::instance()->plateJudge(resultVec1, resultVec2);
  if (result != 0) {
    LOG(ERROR) << "Cannot judge plates\n";
    return -2;
  }
  LOG(INFO) << "Judge plates\n";

  for (size_t i = 0, num = resultVec2.size(); i < num; ++i) {
    imshow(string("plate_judge_") + char('a' + i), resultVec2[i]);
    waitKey(0);
  }

  vector<CPlate> resultVec3;
  CPlateDetect pd;
  pd.setPDLifemode(true);

  result = pd.plateDetect(src, resultVec3);
  if (result != 0) {
    LOG(ERROR) << "Cannot detect plates\n";
    return -3;
  }
  LOG(INFO) << "Detect plates\n";

  for (size_t i = 0, num = resultVec3.size(); i < num; ++i) {
    imshow(string("plate_detect_") + char('a' + i),
           resultVec3[i].getPlateMat());
    waitKey(0);
  }

  const string plate_file_name2 = "resources/image/test_chars_SuEUK722.jpg";
  cv::Mat src2 = cv::imread(plate_file_name2);
  std::vector<cv::Mat> resultVec11;
  CCharsSegment cs;
  result = cs.charsSegment(src2, resultVec11);
  if (result != 0) {
    LOG(ERROR) << "Cannot segemnt chars\n";
    return -5;
  }
  LOG(INFO) << "Segemnt chars\n";

  std::string plateIdenfity;
  for (size_t i = 0, num = resultVec11.size(); i < num; ++i) {
    cv::Mat resultMat = resultVec11[i];
    // std::pair<std::string, std::string> character;
    auto character = CharsIdentify::instance()->identify(resultMat, i == 0);
    plateIdenfity.append(character.second);
    imshow(string("chars_segment_") + char('a' + i), resultMat);
  }
  waitKey(0);
  std::string plateLicense = "苏EUK722";
  std::cout << "[True plateLicense     ] " << plateLicense << std::endl;
  std::cout << "[Idenfity plateLicense ] " << plateIdenfity << std::endl;

  CCharsRecognise cr;
  string plateRecognise = "";
  result = cr.charsRecognise(src2, plateRecognise);
  if (result != 0) {
    LOG(ERROR) << "Cannot recognise chars:\n";
    return -6;
  }
  LOG(INFO) << "Recognise chars:\n";
  std::cout << "charRecognise: " << plateRecognise << std::endl;
  return 0;
}
int test_recognise() {
  std::string svm_model =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/svm_model.xml";
  std::string ann_model_10x10 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_10x10.xml";
  std::string ann_model_20x20 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_20x20.xml";

  CharsIdentify::instance()->LoadModel(ann_model_10x10);
  CPlateJudge::instance()->LoadModel(svm_model);

  LOG(INFO) << "Start test_recognise()" << endl;
  const string plate_file_name = "resources/image/test_plate_SuA0CP56.jpg";
  std::cout << plate_file_name << std::endl;
  Mat src = imread(plate_file_name);
  // vector<string> plateVec;
  vector<CPlate> resultVec;

  CPlateRecognise pr;
  pr.setMaxPlates(4);

  pr.setDetectType(mypr::PR_DETECT_COLOR);
  pr.setDetectType(mypr::PR_DETECT_SOBEL);
  int result = pr.plateRecognize(src, resultVec);
  if (result != 0) {
    LOG(ERROR) << "Cannot recognize plates\n";
    return -4;
  }
  LOG(INFO) << "Recognize plates\n";

  for (size_t i = 0, num = resultVec.size(); i < num; ++i) {
    cout << "plateRecognize: " << resultVec[i].getPlateStr() << std::endl;
  }
}

int i = 0;
std::string::size_type a = 10;
int main(int argc, char *argv[]) {
  // Initialize Google's logging library.
  google::InitGoogleLogging("EasyPR");
  google::SetLogDestination(google::INFO, "/home/auss/Projects/Qt/EasyPR/Log/");
  LOG(INFO) << "Just start gloging...\n";
  if (argc == 2) {
    LOG(INFO) << argv[0] << endl << argv[1] << endl;
    string path = string(argv[1]);
    // changeDirectory(path.substr(0, path.find_last_of(PATH_SEP) + 1).c_str());
    changeDirectory(path.c_str());
  }

  std::string svm_model =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/svm_model.xml";
  std::string ann_model_10x10 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_10x10.xml";
  std::string ann_model_20x20 =
      "/home/auss/Projects/Qt/EasyPR/mypr/resources/train/ann_model_20x20.xml";

  CharsIdentify::instance()->LoadModel(ann_model_10x10);
  CPlateJudge::instance()->LoadModel(svm_model);


  // test_train_model();
  // test_ann();
  test_pr();

  //TODO:fixme!
  //test_recognise();
}
