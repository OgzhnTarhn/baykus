cmake_minimum_required(VERSION 3.24)
project(baykus_engine LANGUAGES CXX)

# Faz 3 tasarımı §4.1: C++17, -Wall -Wextra -Werror; yalnız Docker'da derlenir.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BAYKUS_SANITIZE "AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

find_package(OpenCV 5.0.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(GTest REQUIRED)

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS /opt/onnxruntime/include REQUIRED)
find_library(ORT_LIBRARY onnxruntime PATHS /opt/onnxruntime/lib REQUIRED)
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_INCLUDE_DIR})

add_library(baykus_warnings INTERFACE)
target_compile_options(baykus_warnings INTERFACE -Wall -Wextra -Werror)
if(BAYKUS_SANITIZE)
  target_compile_options(baykus_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(baykus_warnings INTERFACE -fsanitize=address,undefined)
endif()

add_library(baykus_core STATIC
  src/calibration.cpp
  src/detections.cpp
  src/detector.cpp
  src/image_io.cpp
  src/parity.cpp
  src/preprocess.cpp
  src/version.cpp)
target_include_directories(baykus_core PUBLIC include)
target_link_libraries(baykus_core
  PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime Eigen3::Eigen nlohmann_json::nlohmann_json
         OpenSSL::Crypto
  PRIVATE baykus_warnings)

add_executable(baykus_buildinfo apps/buildinfo.cpp)
target_link_libraries(baykus_buildinfo PRIVATE baykus_core baykus_warnings)

enable_testing()
add_library(baykus_test_support STATIC tests/fixtures.cpp)
target_link_libraries(baykus_test_support PUBLIC baykus_core GTest::gtest PRIVATE baykus_warnings)

add_executable(baykus_tests
  tests/smoke_test.cpp
  tests/image_io_test.cpp
  tests/preprocess_test.cpp
  tests/parity_rules_test.cpp
  tests/calibration_test.cpp
  tests/detector_test.cpp)
target_link_libraries(baykus_tests PRIVATE baykus_test_support GTest::gtest_main baykus_warnings)
include(GoogleTest)
gtest_discover_tests(baykus_tests DISCOVERY_MODE PRE_TEST)

add_executable(baykus_parity_report apps/parity_report.cpp)
target_link_libraries(baykus_parity_report PRIVATE baykus_test_support baykus_warnings)

# Fikstür zorunluluğu (BAYKUS_REQUIRE_FIXTURES=1): eksik dosya testi başarısız kılmalı.
add_executable(baykus_fixture_probe tests/fixture_probe.cpp)
target_link_libraries(baykus_fixture_probe PRIVATE baykus_test_support GTest::gtest_main baykus_warnings)
add_test(NAME Fixtures.RequireFailsWhenMissing COMMAND baykus_fixture_probe)
set_tests_properties(Fixtures.RequireFailsWhenMissing PROPERTIES
  ENVIRONMENT "BAYKUS_REQUIRE_FIXTURES=1;BAYKUS_FIXTURES=/nonexistent-fixtures"
  WILL_FAIL TRUE)
add_test(NAME Fixtures.SkipWhenNotRequired COMMAND baykus_fixture_probe)
set_tests_properties(Fixtures.SkipWhenNotRequired PROPERTIES
  ENVIRONMENT "BAYKUS_REQUIRE_FIXTURES=0;BAYKUS_FIXTURES=/nonexistent-fixtures")
