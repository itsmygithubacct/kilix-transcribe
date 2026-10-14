add_executable(kilix-diarize-runner runner.cc)
target_include_directories(kilix-diarize-runner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(kilix-diarize-runner PRIVATE sherpa-onnx-core Eigen3::Eigen)
target_compile_features(kilix-diarize-runner PRIVATE cxx_std_17)
target_compile_definitions(kilix-diarize-runner PRIVATE
  SHERPA_ONNX_ENABLE_DIRECTML=0 SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=1 SHERPA_ONNX_ENABLE_TTS=0)
target_link_options(kilix-diarize-runner PRIVATE -Wl,-z,defs -Wl,-z,now)
set_target_properties(kilix-diarize-runner PROPERTIES
  BUILD_WITH_INSTALL_RPATH YES INSTALL_RPATH "$ORIGIN/lib"
  INSTALL_RPATH_USE_LINK_PATH NO CXX_EXTENSIONS NO)
