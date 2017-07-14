#include "train_ann.hpp"
#include "chars_identify.hpp"
#include "config.hpp"
#include "core_func.hpp"
#include "feature.hpp"
#include "util.hpp"
#include <ctime>
#include <numeric>

namespace mypr {

CTrainAnn::CTrainAnn(const char *chars_folder, const char *ann_xml)
    : m_chars_folder(chars_folder), m_ann_xml(ann_xml) {
  m_ann = cv::ml::ANN_MLP::create();
  type = 0;
}

void CTrainAnn::train() {
  int classNumber = 0;

  cv::Mat layers;

  int input_number = 0;
  int hidden_number = 0;
  int output_number = 0;

  if (type == 0) {
    classNumber = kCharsTotalNumber;

    input_number = kAnnInput;
    hidden_number = kNeurons;
    output_number = classNumber;
  } else if (type == 1) {
    classNumber = kChineseNumber;

    input_number = kAnnInput;
    hidden_number = kNeurons;
    output_number = classNumber;
  }

  int N = input_number;
  int m = output_number;
  int first_hidden_neurons =
      int(std::sqrt((m + 2) * N) + 2 * std::sqrt(N / (m + 2)));
  int second_hidden_neurons = int(m * std::sqrt(N / (m + 2)));

  bool useTLFN = true;
  if (!useTLFN) {
    layers.create(1, 3, CV_32SC1);
    layers.at<int>(0) = input_number;
    layers.at<int>(1) = hidden_number;
    layers.at<int>(2) = output_number;
  } else {
    LOG(INFO) << ">> Use two-layers neural networks,\n";
    LOG(INFO) << ">> First_hidden_neurons: " << first_hidden_neurons
              << std::endl;
    LOG(INFO) << ">> Second_hidden_neurons: " << second_hidden_neurons
              << std::endl;

    layers.create(1, 4, CV_32SC1);
    layers.at<int>(0) = input_number;
    layers.at<int>(1) = first_hidden_neurons;
    layers.at<int>(2) = second_hidden_neurons;
    layers.at<int>(3) = output_number;
  }

  m_ann->setLayerSizes(layers);
  m_ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
  m_ann->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
  m_ann->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 10000, 0.001));
  m_ann->setBackpropWeightScale(0.1);
  m_ann->setBackpropMomentumScale(0.1);

  // using raw data or raw + synthic data.
  // auto traindata = tdata();
  auto traindata = sdata(350);

  long start = Util::getTimestamp();
  m_ann->train(traindata);
  long end = Util::getTimestamp();
  m_ann->save(m_ann_xml);

  LOG(INFO) << "Testing ANN model, please wait..." << std::endl;
  google::FlushLogFiles(google::GLOG_INFO);
  test();
  LOG(INFO) << "Your ANN Model was saved to " << m_ann_xml << std::endl;
  LOG(INFO) << "Training done. Time elapse: " << (end - start) / (1000 * 60)
            << "minute" << std::endl;
}

std::pair<std::string, std::string> CTrainAnn::identifyChinese(cv::Mat input) {
  cv::Mat feature = charFeatures(input, kPredictSize);
  float maxVal = -2;
  int result = -1;

  cv::Mat output(1, kChineseNumber, CV_32FC1);
  m_ann->predict(feature, output);

  for (int j = kCharactersNumber; j < kCharsTotalNumber; ++j) {
    float val = output.at<float>(j);
    if (val > maxVal) {
      maxVal = val;
      result = j;
    }
  }

  auto index = result + kCharactersNumber;
  std::string key = kChars[index];
  std::string province = CHARS_DICT.at(key);

  return std::make_pair(key, province);
}

std::pair<std::string, std::string> CTrainAnn::identify(cv::Mat input) {
  cv::Mat feature = charFeatures(input, kPredictSize);
  float maxVal = -2;
  int result = -1;

  cv::Mat output(1, kCharsTotalNumber, CV_32FC1);
  m_ann->predict(feature, output);

  for (int j = 0; j < kCharsTotalNumber; ++j) {
    float val = output.at<float>(j);
    if (val > maxVal) {
      maxVal = val;
      result = j;
    }
  }

  auto index = result;
  std::string key = kChars[index];
  std::string value = CHARS_DICT.at(key);

  return std::make_pair(key, value);
}

void CTrainAnn::test() {
  assert(!m_chars_folder.empty());
  int corrects_all = 0, sum_all = 0;
  std::vector<float> rate_list;

  int start = 0, end = kCharsTotalNumber;
  int classNumber = 0;
  if (type == 0) {
    classNumber = kCharsTotalNumber;
    start = 0;
  }
  if (type == 1) {
    classNumber = kChineseNumber;
    start = kCharactersNumber;
  }
  for (int i = start; i < end; ++i) {
    auto char_key = kChars[i];
    char sub_folder[512] = {0};

    sprintf(sub_folder, "%s/%s", m_chars_folder.c_str(), char_key.c_str());
    LOG(INFO) << ">> Testing characters " << char_key << " in " << sub_folder
              << "\n";

    auto chars_files = Util::getFiles(sub_folder);
    int corrects = 0, sum = 0;
    std::vector<std::pair<std::string, std::string>> error_files;

    for (auto file : chars_files) {
      auto img = cv::imread(file, 0); // a grayscale image
      std::pair<std::string, std::string> ch;

      if (type == 0) {
        ch = identify(img);
      }
      if (type == 1) {
        ch = identifyChinese(img);
      }

      if (ch.first == char_key) {
        ++corrects;
        ++corrects_all;
      } else {
        error_files.push_back(
            std::make_pair(Util::getFileName(file), ch.second));
      }
      ++sum;
      ++sum_all;
    }
    float rate = (float)corrects / (sum == 0 ? 1 : sum);
    fprintf(stdout, ">>   [sum: %d, correct: %d, rate: %.2f]\n", sum, corrects,
            rate);
    rate_list.push_back(rate);

    std::string error_string;
    auto end = error_files.end();
    if (error_files.size() >= 10) {
      end -= static_cast<size_t>(error_files.size() * (1 - 0.1));
    }
    for (auto k = error_files.begin(); k != end; ++k) {
      auto kv = *k;
      error_string.append("\n       ").append(kv.first).append(": ").append(
          kv.second);
      if (k != end - 1) {
        error_string.append(",\n");
      } else {
        error_string.append("\n       ...");
      }
    }
    LOG(INFO) << ">>  " << error_string << std::endl << std::endl;
  }
  LOG(INFO) << ">>   [sum_all: " << sum_all << ", correct_all: " << corrects_all
            << ", rate: " << (float)corrects_all / (sum_all == 0 ? 1 : sum_all)
            << "]\n";

  double rate_sum = std::accumulate(rate_list.begin(), rate_list.end(), 0.0);
  double rate_mean = rate_sum / (rate_list.size() == 0 ? 1 : rate_list.size());

  LOG(INFO) << ">>   [classNumber: " << classNumber
            << ", avg_rate: " << rate_mean << "]\n";
}

cv::Mat getSyntheticImage(const Mat &image) {
  int rand_type = rand();
  Mat result = image.clone();

  if (rand_type % 2 == 0) {
    int ran_x = rand() % 5 - 2;
    int ran_y = rand() % 5 - 2;

    result = translateImg(result, ran_x, ran_y);
  } else if (rand_type % 2 != 0) {
    float angle = float(rand() % 15 - 7);

    result = rotateImg(result, angle);
  }

  return result;
}

cv::Ptr<cv::ml::TrainData> CTrainAnn::sdata(size_t number_for_count) {
  assert(!m_chars_folder.empty());

  cv::Mat samples;
  std::vector<int> labels;
  srand((unsigned)time(0));

  int start = 0, end = kCharsTotalNumber;
  int classNumber = 0;
  if (type == 0) {
    classNumber = kCharsTotalNumber;
    start = 0;
  }
  if (type == 1) {
    classNumber = kChineseNumber;
    start = kCharactersNumber;
  }
  for (int i = start; i < end; ++i) {
    LOG(INFO) << "[" << i << ":" << end << "]" << std::endl;

    auto char_key = kChars[i];
    char sub_folder[512] = {0};

    sprintf(sub_folder, "%s/%s", m_chars_folder.c_str(), char_key.c_str());
    LOG(INFO) << ">> Testing characters " << char_key << " in " << sub_folder
              << "\n";

    auto chars_files = Util::getFiles(sub_folder);
    size_t char_size = chars_files.size();
    LOG(INFO) << ">> Characters count: " << char_size << "\n";

    std::vector<cv::Mat> matVec;
    matVec.reserve(number_for_count);
    for (auto file : chars_files) {
      auto img = cv::imread(file, 0); // a grayscale image
      matVec.push_back(img);
    }

    for (int t = 0; t < (int)number_for_count - (int)char_size; t++) {
      int rand_range = char_size + t;
      int ran_num = rand() % rand_range;
      auto img = matVec.at(ran_num);
      auto simg = getSyntheticImage(img);
      matVec.push_back(simg);
      if (1) {
        std::stringstream ss(std::stringstream::in | std::stringstream::out);
        ss << sub_folder << "/" << i << "_" << t << "_" << ran_num << ".jpg";
        imwrite(ss.str(), simg);
      }
    }

    LOG(INFO) << ">> After Synthetic " << i
              << " , characters count: " << matVec.size() << std::endl;
    google::FlushLogFiles(google::GLOG_INFO);

    for (auto img : matVec) {
      auto feature = charFeatures(img, kPredictSize);

      samples.push_back(feature);
      labels.push_back(i);
    }
  }

  cv::Mat samples_;
  samples.convertTo(samples_, CV_32F);
  cv::Mat train_classes =
      cv::Mat::zeros((int)labels.size(), classNumber, CV_32F);

  for (int i = 0; i < train_classes.rows; ++i) {
    train_classes.at<float>(i, labels[i]) = 1.f;
  }

  return cv::ml::TrainData::create(samples_, cv::ml::SampleTypes::ROW_SAMPLE,
                                   train_classes);
}

cv::Ptr<cv::ml::TrainData> CTrainAnn::tdata() {
  assert(!m_chars_folder.empty());

  cv::Mat samples;
  std::vector<int> labels;

  std::cout << "Collecting chars in " << m_chars_folder << std::endl;

  int start = 0, end = kCharsTotalNumber;
  int classNumber = 0;
  if (type == 0) {
    classNumber = kCharsTotalNumber;
    start = 0;
  }
  if (type == 1) {
    classNumber = kChineseNumber;
    start = kCharactersNumber;
  }
  for (int i = start; i < end; ++i) {
    auto char_key = kChars[i];
    char sub_folder[512] = {0};

    sprintf(sub_folder, "%s/%s", m_chars_folder.c_str(), char_key.c_str());
    LOG(INFO) << "  >> Featuring characters " << char_key << " in "
              << sub_folder << std::endl;

    auto chars_files = Util::getFiles(sub_folder);
    for (auto file : chars_files) {
      auto img = cv::imread(file, 0); // a grayscale image
      auto fps = charFeatures(img, kPredictSize);

      samples.push_back(fps);
      labels.push_back(i);
    }
  }

  cv::Mat samples_;
  samples.convertTo(samples_, CV_32F);
  cv::Mat train_classes =
      cv::Mat::zeros((int)labels.size(), classNumber, CV_32F);

  for (int i = 0; i < train_classes.rows; ++i) {
    train_classes.at<float>(i, labels[i]) = 1.f;
  }

  return cv::ml::TrainData::create(samples_, cv::ml::SampleTypes::ROW_SAMPLE,
                                   train_classes);
}
}
