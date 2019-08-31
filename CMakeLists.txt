cmake_minimum_required(VERSION 3.9)
project(fastwfc VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()

include(GNUInstallDirs)

set(SOURCE_FILES 
src/include/fastwfc/wfc.hpp)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

set(LIBRARY_OUTPUT_PATH lib CACHE PATH "Build directory" FORCE)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1)

add_executable(${PROJECT_NAME}_demo examples/src/lib/main.cpp)
set_target_properties(fastwfc PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME}_demo ${PROJECT_NAME}_static)

target_include_directories(${PROJECT_NAME}_demo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/examples/src/include>)

install(TARGETS ${PROJECT_NAME} EXPORT FastWFCConfig
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS ${PROJECT_NAME}_demo EXPORT FastWFCConfig
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME  DESTINATION share/fastwfc-examples)

install(DIRECTORY examples/samples DESTINATION share/fastwfc-examples)
install(FILES examples/samples.xml DESTINATION share/fastwfc-examples)
install(FILES examples/results/.empty DESTINATION share/fastwfc-examples/results)

install(DIRECTORY src/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT FastWFCConfig DESTINATION share/fastwfc/cmake)

export(TARGETS ${PROJECT_NAME} FILE FastWFCConfig.cmake)
