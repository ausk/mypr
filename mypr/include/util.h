#ifndef UTIL_HPP
#define UTIL_HPP

#include <glog/logging.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#if defined(WIN32) || defined(_WIN32)
#define OS_WINDOWS
#elif defined(__APPLE__) || defined(APPLE)
#define OS_UNIX
#elif defined(__linux__) || defined(linux)
#define OS_LINUX
#endif

#ifdef OS_WINDOWS
#include <ctime>
#include <dirent.h>
#include <io.h>
#include <windows.h>
#define cd _chdir
#define mkdir _mkdir
#define access _access
#define cwd _getcwd
#define PATH_DELIMITER '\\'
#else
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#define cd chdir
#define cwd getcwd
#define PATH_DELIMITER '/'
#endif

#ifdef OS_UNIX
#include <sys/timeb.h>
#endif

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)                                                        \
  if ((p)) {                                                                   \
    delete (p);                                                                \
    (p) = NULL;                                                                \
  }
#endif

namespace Util {
std::string getCurrWorkingDirectory();
bool changeDirectory(const std::string &folder);
std::vector<std::string> splitString(const std::string &str,
                                     const char delimiter);
std::vector<std::string> getFiles(const std::string &folder, bool all = true);

std::string getFileName(const std::string &path, bool postfix = false);
long getTimestamp();
bool mkdir(const std::string folder);
bool imwrite(const std::string &file, const cv::Mat &image);

#ifndef True
#define True true
#endif
#ifndef TRUE
#define TRUE true
#endif

#ifndef False
#define False false
#endif
#ifndef FALSE
#define FALSE false
#endif

#ifdef max
#undef max
template <typename T> T max(T x, T y) { return x > y ? x : y; }
#endif
#ifdef min
#undef min
template <typename T> T min(T x, T y) { return x < y ? x : y; }
#endif

#ifdef abs
#undef abs
template <typename T> T abs(T x) { return x < 0 ? -x : x; }
#endif
}  /*! namespace mypr */

#endif // UTIL_HPP
