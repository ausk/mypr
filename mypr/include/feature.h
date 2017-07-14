#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <opencv2/opencv.hpp>
namespace mypr {

//! 获得车牌的特征数

cv::Mat getHistogram(cv::Mat in);

//! EasyPR的getFeatures回调函数
//! 用于从车牌的image生成svm的训练特征features

typedef void (*svmCallback)(const cv::Mat &image, cv::Mat &features);

//!  EasyPR的getFeatures回调函数
//! 本函数是获取垂直和水平的直方图图值

void getHistogramFeatures(const cv::Mat &image, cv::Mat &features);

void getSIFTFeatures(const cv::Mat &image, cv::Mat &features);
void getHOGFeatures(const cv::Mat &image, cv::Mat &features);
void getHSVHistFeatures(const cv::Mat &image, cv::Mat &features);
void getLBPFeatures(const cv::Mat &image, cv::Mat &features);

void getLBPplusHistFeatures(const cv::Mat &image, cv::Mat &features);

//! get character feature
cv::Mat charFeatures(const cv::Mat &in, int sizeData);

} /*! \namespace mypr*/
#endif // FEATURE_HPP
