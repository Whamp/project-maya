# Optional, pinned host-staged client. System dependencies are Mooncake's own
# documented build dependencies; local Maya builds never fetch or need them.
option(MAYA_BUILD_MOONCAKE_CLIENT "Provide a build target for the pinned FlashStates Mooncake client/master" OFF)
if(MAYA_BUILD_MOONCAKE_CLIENT)
  if(NOT UNIX)
    message(FATAL_ERROR "The optional Mooncake client target requires a supported Unix host; local state caching works on Windows")
  endif()
  include(ExternalProject)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  ExternalProject_Add(maya-mooncake-client
    EXCLUDE_FROM_ALL TRUE
    PREFIX ${CMAKE_BINARY_DIR}/mooncake-client
    GIT_REPOSITORY https://github.com/FutureMLS-Lab/Mooncake.git
    GIT_TAG 59ca03f3c44131cf0ecea299d05bb8ea31f2032f
    GIT_SUBMODULES extern/pybind11
    UPDATE_DISCONNECTED TRUE
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG_SYMBOLS=OFF
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE} -DWITH_STORE_RUST=OFF
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=$ORIGIN
      -DUSE_CUDA=OFF -DUSE_ETCD=OFF -DBUILD_UNIT_TESTS=OFF
      -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=OFF
      -DYLT_BUILD_EXAMPLES=OFF -DYLT_BUILD_BENCHMARK=OFF -DYLT_BUILD_TESTS=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target mooncake_master store --parallel 2
    INSTALL_COMMAND ${CMAKE_COMMAND} -DCLIENT_BUILD=<BINARY_DIR>
      -DCLIENT_DEST=${CMAKE_BINARY_DIR}/mooncake-client/runtime
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_mooncake_state.cmake)
endif()
