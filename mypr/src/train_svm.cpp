#include "train_svm.hpp"
#include "core_func.hpp"
#include "feature.hpp"
#include "util.hpp"

namespace mypr {
CTrainSvm::CTrainSvm(const char *plates_folder, const char *svm_xml)
    : m_plates_folder(plates_folder), m_svm_xml(svm_xml) {
  assert(!m_plates_folder.empty());
  assert(!m_svm_xml.empty());
}

void CTrainSvm::train() {
  m_svm = cv::ml::SVM::create();
  m_svm->setType(cv::ml::SVM::C_SVC);
  // m_svm->setKernel(cv::ml::SVM::CH2);
  m_svm->setDegree(0.1);
  m_svm->setGamma(0.1);
  m_svm->setCoef0(0.1);
  m_svm->setC(1);
  m_svm->setNu(0.1);
  m_svm->setP(0.1);
  m_svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));

  auto train_data = tdata();

  fprintf(stdout, ">> Training SVM model, please wait...\n");
  long start = Util::getTimestamp();
  // m_svm->trainAuto(train_data, 10, SVM::getDefaultGrid(SVM::C),
  //                SVM::getDefaultGrid(SVM::GAMMA),
  //                SVM::getDefaultGrid(SVM::P),
  //                SVM::getDefaultGrid(SVM::NU),
  //                SVM::getDefaultGrid(SVM::COEF),
  //                SVM::getDefaultGrid(SVM::DEGREE), true);
  m_svm->train(train_data);

  long end = Util::getTimestamp();
  LOG(INFO) << ">> Training done. Time elapse: " << end - start << "ms\n"
            << ">> Saving model file...\n";

  m_svm->save(m_svm_xml);

  LOG(INFO) << ">> Your SVM Model was saved to " << m_svm_xml << std::endl
            << ">> Testing...\n";

  this->test();
}

void CTrainSvm::test() {
  // 1.4 bug fix: old 1.4 ver there is no null judge
  // if (NULL == svm_)
  m_svm = cv::Algorithm::load<cv::ml::SVM>(m_svm_xml);

  if (m_test_file_list.empty()) {
    this->prepare();
  }

  double count_all = m_test_file_list.size();
  double ptrue_rtrue = 0;
  double ptrue_rfalse = 0;
  double pfalse_rtrue = 0;
  double pfalse_rfalse = 0;

  for (auto item : m_test_file_list) {
    auto image = cv::imread(item.filename);
    if (!image.data) {

      std::cout << "no" << std::endl;
      continue;
    }
    cv::Mat feature;
    getLBPFeatures(image, feature);

    auto predict = int(m_svm->predict(feature));
    // std::cout << "predict: " << predict << std::endl;

    auto real = item.label;
    if (predict == kForward && real == kForward) {
      ptrue_rtrue++;
    }
    if (predict == kForward && real == kInverse) {
      ptrue_rfalse++;
    }
    if (predict == kInverse && real == kForward) {
      pfalse_rtrue++;
    }
    if (predict == kInverse && real == kInverse) {
      pfalse_rfalse++;
    }
  }

  std::cout << "count_all: " << count_all << std::endl;
  std::cout << "ptrue_rtrue: " << ptrue_rtrue << std::endl;
  std::cout << "ptrue_rfalse: " << ptrue_rfalse << std::endl;
  std::cout << "pfalse_rtrue: " << pfalse_rtrue << std::endl;
  std::cout << "pfalse_rfalse: " << pfalse_rfalse << std::endl;

  double precise = 0;
  if (ptrue_rtrue + ptrue_rfalse != 0) {
    precise = ptrue_rtrue / (ptrue_rtrue + ptrue_rfalse);
    std::cout << "precise: " << precise << std::endl;
  } else {
    std::cout << "precise: "
              << "NA" << std::endl;
  }

  double recall = 0;
  if (ptrue_rtrue + pfalse_rtrue != 0) {
    recall = ptrue_rtrue / (ptrue_rtrue + pfalse_rtrue);
    std::cout << "recall: " << recall << std::endl;
  } else {
    std::cout << "recall: "
              << "NA" << std::endl;
  }

  double Fsocre = 0;
  if (precise + recall != 0) {
    Fsocre = 2 * (precise * recall) / (precise + recall);
    std::cout << "Fsocre: " << Fsocre << std::endl;
  } else {
    std::cout << "Fsocre: "
              << "NA" << std::endl;
  }
}

void CTrainSvm::prepare() {
  srand(unsigned(time(NULL)));

  char buffer[260] = {0};

  sprintf(buffer, "%s/has/train", m_plates_folder.c_str());
  auto has_file_train_list = Util::getFiles(buffer);
  std::random_shuffle(has_file_train_list.begin(), has_file_train_list.end());

  sprintf(buffer, "%s/has/test", m_plates_folder.c_str());
  auto has_file_test_list = Util::getFiles(buffer);
  std::random_shuffle(has_file_test_list.begin(), has_file_test_list.end());

  sprintf(buffer, "%s/no/train", m_plates_folder.c_str());
  auto no_file_train_list = Util::getFiles(buffer);
  std::random_shuffle(no_file_train_list.begin(), no_file_train_list.end());

  sprintf(buffer, "%s/no/test", m_plates_folder.c_str());
  auto no_file_test_list = Util::getFiles(buffer);
  std::random_shuffle(no_file_test_list.begin(), no_file_test_list.end());

  LOG(INFO) << ">> Collecting train data...\n";

  for (auto file : has_file_train_list) {
    m_train_file_list.push_back({file, kForward});
  }

  for (auto file : no_file_train_list) {
    m_train_file_list.push_back({file, kInverse});
  }

  LOG(INFO) << ">> Collecting test data...\n";

  for (auto file : has_file_test_list) {
    m_test_file_list.push_back({file, kForward});
  }

  for (auto file : no_file_test_list) {
    m_test_file_list.push_back({file, kInverse});
  }

  LOG(INFO) << ">> Total size: "<<m_train_file_list.size()<<std::endl;
}

cv::Ptr<cv::ml::TrainData> CTrainSvm::tdata() {
  this->prepare();

  cv::Mat samples;
  std::vector<int> responses;

  for (auto f : m_train_file_list) {
    auto image = cv::imread(f.filename);
    if (!image.data) {
      LOG(INFO) << ">> Invalid image: " << f.filename << " ignore.\n";
      continue;
    }
    cv::Mat feature;
    getLBPFeatures(image, feature);
    feature = feature.reshape(1, 1);

    samples.push_back(feature);
    responses.push_back(int(f.label));
  }

  cv::Mat samples_, responses_;
  samples.convertTo(samples_, CV_32FC1);
  cv::Mat(responses).copyTo(responses_);

  return cv::ml::TrainData::create(samples_, cv::ml::SampleTypes::ROW_SAMPLE,
                                   responses_);
}
}
