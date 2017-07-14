#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <map>
#include <memory>
#include <string>

namespace mypr {

static bool kDebug = false;

enum Color { BLUE, YELLOW, WHITE, UNKNOWN };

enum LocateType { SOBEL, COLOR, CMSER, OTHER };

enum CharSearchDirection { LEFT, RIGHT };

enum {
  PR_DETECT_SOBEL = (1 << 0), /**Sobel detect type, using twice Sobel  */
  PR_DETECT_COLOR = (1 << 1), /**Color detect type   */
  PR_DETECT_CMSER = (1 << 2), /**Character detect type, using mser  */
};

static const char *kDefaultSvmPath = "resources/model/svm.xml";
static const char *kLBPSvmPath = "resources/model/svm_lbp_final.xml";
static const char *kDefaultAnnPath = "resources/model/ann.xml";
static const char *kChineseAnnPath = "resources/model/ann_chinese.xml";

typedef enum {
  kForward = 1, // correspond to "has plate"
  kInverse = 0  // correspond to "no plate"
} SvmLabel;

static const int kPlateResizeWidth = 136;
static const int kPlateResizeHeight = 36;

static const int kShowWindowWidth = 800;
static const int kShowWindowHeight = 600;

static const float kSvmPercentage = 0.7f;


static const int kCharacterSize = 10; //10
static const int kChineseSize = 20;
static const int kPredictSize = kCharacterSize;

static const int kCharacterInput = kCharacterSize*(kCharacterSize+2);
static const int kChineseInput = kChineseSize*(kChineseSize+2);   // 20*20 + 20*2
static const int kAnnInput = kPredictSize*(kPredictSize+2);

static const int kNeurons = 40;

static const int kCharactersNumber = 34;
static const int kChineseNumber = 31;
static const int kCharsTotalNumber = 65;

static const std::string kChars[] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    /*  10  */
    "A", "B", "C", "D", "E", "F", "G", "H", /* {"I", "I"} */
    "J", "K", "L", "M", "N",                /* {"O", "O"} */
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    /*  24  */
    "zh_cuan", "zh_e", "zh_gan", "zh_gan1", "zh_gui", "zh_gui1", "zh_hei",
    "zh_hu", "zh_ji", "zh_jin", "zh_jing", "zh_jl", "zh_liao", "zh_lu",
    "zh_meng", "zh_min", "zh_ning", "zh_qing", "zh_qiong", "zh_shan", "zh_su",
    "zh_sx", "zh_wan", "zh_xiang", "zh_xin", "zh_yu", "zh_yu1", "zh_yue",
    "zh_yun", "zh_zang", "zh_zhe"
    /*  31  */
};

static const std::map<std::string, std::string> CHARS_DICT = {

    {"0", "0"},
    {"1", "1"},
    {"2", "2"},
    {"3", "3"},
    {"4", "4"},
    {"5", "5"},
    {"6", "6"},
    {"7", "7"},
    {"8", "8"},
    {"9", "9"},
    /*  10  */
    {"A", "A"},
    {"B", "B"},
    {"C", "C"},
    {"D", "D"},
    {"E", "E"},
    {"F", "F"},
    {"G", "G"},
    {"H", "H"},
    /*  I   */
    {"J", "J"},
    {"K", "K"},
    {"L", "L"},
    {"M", "M"},
    {"N", "N"},
    /*  O   */
    {"P", "P"},
    {"Q", "Q"},
    {"R", "R"},
    {"S", "S"},
    {"T", "T"},
    {"U", "U"},
    {"V", "V"},
    {"W", "W"},
    {"X", "X"},
    {"Y", "Y"},
    {"Z", "Z"},
    /*  24  */
    {"zh_cuan", "川"},
    {"zh_e", "鄂"},
    {"zh_gan", "赣"},
    {"zh_gan1", "甘"},
    {"zh_gui", "贵"},
    {"zh_gui1", "桂"},
    {"zh_hei", "黑"},
    {"zh_hu", "沪"},
    {"zh_ji", "冀"},
    {"zh_jin", "津"},
    {"zh_jing", "京"},
    {"zh_jl", "吉"},
    {"zh_liao", "辽"},
    {"zh_lu", "鲁"},
    {"zh_meng", "蒙"},
    {"zh_min", "闽"},
    {"zh_ning", "宁"},
    {"zh_qing", "青"},
    {"zh_qiong", "琼"},
    {"zh_shan", "陕"},
    {"zh_su", "苏"},
    {"zh_sx", "晋"},
    {"zh_wan", "皖"},
    {"zh_xiang", "湘"},
    {"zh_xin", "新"},
    {"zh_yu", "豫"},
    {"zh_yu1", "渝"},
    {"zh_yue", "粤"},
    {"zh_yun", "云"},
    {"zh_zang", "藏"},
    {"zh_zhe", "浙"},
    /*  31  */

};

}

#endif // CONFIG_HPP
