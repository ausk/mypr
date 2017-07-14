/*
 * 2017.01.06 13:40:56 CST
 */

#ifndef _MY_PREPROCESS_HPP
#define _MY_PREPROCESS_HPP
#include <opencv2/opencv.hpp>
namespace mypr {
namespace preprocess {
cv::Mat maskFace(cv::Mat &img, double scale);
}
cv::Mat blurImage(cv::Mat&img);
}

#endif //_MY_PREPROCESS_HPP
