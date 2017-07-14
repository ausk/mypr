#include "mypreprocess.hpp"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;
namespace mypr {
namespace preprocess {
cv::Mat maskFace(cv::Mat &img, double scale) {
  cv::CascadeClassifier cascade;
  std::string cascadeName =
      "resources/model/haarcascade_frontalface_default.xml";

  if (!cascade.load(cascadeName)) {
    LOG(ERROR) << "ERROR: Could not load classifier cascade: \n\t"
               << cascadeName << std::endl;
    return img;
  }

  std::vector<cv::Rect> faces;
  Size sz(cvRound(img.rows / scale), cvRound(img.cols / scale));
  cv::Mat gray;
  cv::Mat smallImg(sz, CV_8UC1);

  cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  resize(gray, smallImg, sz, 0, 0, cv::INTER_LINEAR);
  equalizeHist(smallImg, smallImg);

  cascade.detectMultiScale(smallImg, faces, 1.1, 2,
                           0
                               //|CASCADE_FIND_BIGGEST_OBJECT
                               //|CASCADE_DO_ROUGH_SEARCH
                               | cv::CASCADE_SCALE_IMAGE,
                           cv::Size(30, 30));
  for (auto r = faces.begin(); r != faces.end(); r++) {
    cv::Rect facerect = *r;
    cv::Mat roi =
        img(cv::Rect_<double>(facerect.x * scale, facerect.y * scale,
                              facerect.width * scale, facerect.height * scale));

    // MASK
    int W = 18;
    int H = 18;
    for (int i = W; i < roi.cols; i += W) {
      for (int j = H; j < roi.rows; j += H) {
        uchar s = roi.at<uchar>(j - H / 2, (i - W / 2) * 3);
        uchar s1 = roi.at<uchar>(j - H / 2, (i - W / 2) * 3 + 1);
        uchar s2 = roi.at<uchar>(j - H / 2, (i - W / 2) * 3 + 2);
        for (int ii = i - W; ii <= i; ii++) {
          for (int jj = j - H; jj <= j; jj++) {
            roi.at<uchar>(jj, ii * 3 + 0) = s;
            roi.at<uchar>(jj, ii * 3 + 1) = s1;
            roi.at<uchar>(jj, ii * 3 + 2) = s2;
          }
        }
      }
    }
  }

  return img;
}

cv::Mat blurImage(const cv::Mat &img) {
  int width = img.size().width;
  int height = img.size().height;

  cv::Rect_<double> rect(width * 0.01, height * 0.01, width * 0.99,
                         height * 0.99);
  cv::Mat dst = img(rect);
  GaussianBlur(dst,dst,Size(3,3),0,0,BORDER_DEFAULT);

  return dst;
}
}
}
