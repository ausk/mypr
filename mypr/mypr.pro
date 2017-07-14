QT += core
QT -= gui

CONFIG += c++11

TARGET = mypr
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    mypreprocess.cpp \
    util.cpp \
    train_svm.cpp \
    train_ann.cpp \
    thirdparty/LBP/helper.cpp \
    thirdparty/LBP/lbp.cpp \
    thirdparty/mser/mser2.cpp \
    thirdparty/textDetect/erfilter.cpp \
    thirdparty/xmlParser/xmlParser.cpp \
    feature.cpp \
    chars_identify.cpp \
    core_func.cpp \
    params.cpp \
    plate_locate.cpp \
    plate_judge.cpp \
    plate_detect.cpp \
    chars_segment.cpp \
    chars_recognise.cpp \
    plate_recognise.cpp \
    src/chars_identify.cpp \
    src/chars_recognise.cpp \
    src/chars_segment.cpp \
    src/core_func.cpp \
    src/feature.cpp \
    src/kv.cpp \
    src/main.cpp \
    src/mypreprocess.cpp \
    src/params.cpp \
    src/plate_detect.cpp \
    src/plate_judge.cpp \
    src/plate_locate.cpp \
    src/plate_recognise.cpp \
    src/train_ann.cpp \
    src/train_svm.cpp \
    src/util.cpp


HEADERS += \
    mypreprocess.hpp \
    util.hpp \
    train_svm.hpp \
    train_ann.hpp \
    train.hpp \
    config.hpp \
    thirdparty/LBP/helper.hpp \
    thirdparty/LBP/lbp.hpp \
    thirdparty/mser/mser2.hpp \
    thirdparty/textDetect/erfilter.hpp \
    thirdparty/xmlParser/xmlParser.h \
    character.hpp \
    plate.hpp \
    params.hpp \
    feature.hpp \
    chars_identify.hpp \
    core_func.hpp \
    plate_locate.hpp \
    plate_judge.hpp \
    plate_detect.hpp \
    chars_segment.hpp \
    chars_recognise.hpp \
    plate_recognise.hpp \
    include/character.hpp \
    include/chars_identify.hpp \
    include/chars_recognise.hpp \
    include/chars_segment.hpp \
    include/config.hpp \
    include/core_func.hpp \
    include/feature.hpp \
    include/kv.hpp \
    include/mypreprocess.hpp \
    include/params.hpp \
    include/plate.hpp \
    include/plate_detect.hpp \
    include/plate_judge.hpp \
    include/plate_locate.hpp \
    include/plate_recognise.hpp \
    include/provinces_mapping.hpp \
    include/train.hpp \
    include/train_ann.hpp \
    include/train_svm.hpp \
    include/util.hpp


####### Third part library ######
LIBS += -lpthread -lglog

####### OpenCV #########
INCLUDEPATH += .\
    /usr/local/include\
    /home/auss/Programs/OpenCV/opencv3/include\
    /home/auss/Programs/OpenCV/opencv3/include/opencv\
    /home/auss/Programs/OpenCV/opencv3/include/opencv2

LIBS += /home/auss/Programs/OpenCV/opencv3/lib/libopencv*.so
LIBS += -lglog

DISTFILES += \
    thirdparty/thirdparty.cbp \
    thirdparty/xmlParser/AFPL-license.txt \
    thirdparty/CMakeLists.txt
