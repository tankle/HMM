cmake_minimum_required(VERSION 3.2)
project(HMM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES hmm.cpp)
add_executable(HMM ${SOURCE_FILES} HMM.cpp example/HMMSegment.h example/HMMSegment.cpp example/HMMSegment.cpp example/HMMSegment.h)