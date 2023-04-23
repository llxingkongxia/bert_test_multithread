cmake_minimum_required (VERSION 3.4.1)

project(bert-cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
option(PROFILE_ENABLE "Enable profile" OFF)
option(PROFILE_ENABLE_TEST "Enable profile" OFF)
file(GLOB DIR_SRCS src/*.cc)
file(GLOB UTILS_SRCS utils/*.cc)

if(PROFILE_ENABLE_TEST)
     add_definitions(-DPRFILE_FUNCTION)
     message("build profile binary")
     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
     SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
     SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")
     SET(CMAKE_CXX_FLAGS_RELEASE "-O0")
     SET(CMAKE_CXX_FLAGS_DEBUG "-O0")
endif(PROFILE_ENABLE)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0")

#include(FindProtobuf)

set(Protobuf_LIBRARY /usr/local/lib/libprotobuf.so)
set(Protobuf_INCLUDE_DIR /usr/local/include)
find_package(Protobuf REQUIRED)

#set(MKL_ROOT /opt/intel/mkl)
#include(FindMKL.cmake)
#find_package(MKL REQUIRED)

find_package(Threads REQUIRED)

set(utf8proc_LIBRARY /home/liulei/miniconda2/envs/bert/lib/libutf8proc.so)
set(utf8proc_INCLUDE_DIR /home/liulei/miniconda2/envs/bert/include)

### setting hygonblis #####
#set(BLIS_LIBRARY /home/liulei/hygonblis-dev/lib/libblis-hg.so)
set(BLIS_LIBRARY /home/liulei/hygonblis-dev/lib/libblis.so)
set(BLIS_INCLUDE_DIR /home/liulei/hygonblis-dev/include/blis)

include_directories(${Protobuf_INCLUDE_DIR} ${utf8proc_INCLUDE_DIR} ${BLIS_INCLUDE_DIR})
message("11111${Protobuf_LIBRARY}")
message("22222${Protobuf_INCLUDE_DIR}")

add_library(bertcpp SHARED ${DIR_SRCS} ${UTILS_SRCS})
target_link_libraries(bertcpp ${Protobuf_LIBRARY} ${utf8proc_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${BLIS_LIBRARY})

add_executable(profiletest profile.cpp)
target_link_libraries(profiletest bertcpp)

#enable_testing()
#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIR})

#file(GLOB TEST_SRCS test/*.cpp)
#add_executable(bert_test ${TEST_SRCS})
#target_link_libraries(bert_test bertcpp ${GTEST_BOTH_LIBRARIES})
#add_test(unitest bert_test)
