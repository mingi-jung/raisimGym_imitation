cmake_minimum_required(VERSION 3.5)
project(raisim_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
# SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
# SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)

####################
### dependencies ###
####################
set(Dependencies)

if (UNIX AND NOT APPLE AND NOT ANDROID AND NOT WEBGL AND NOT WIN32)
    set(RAISIM_OS linux)
    list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})
elseif(APPLE)
    set(RAISIM_OS mac)
    list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})
    list(APPEND Dependencies OpenMP::OpenMP_CXX)
elseif(WIN32)
    set(RAISIM_OS win32)
    # list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS}/mt_release)
    list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS}/mt_debug)
    list(APPEND CMAKE_PREFIX_PATH ../thirdParty)
    list(APPEND Dependencies Ws2_32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
endif()

# find_package(raisim CONFIG REQUIRED)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#==
# Dependencies
#==

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})

if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/win32/mt_release/bin")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/win32/mt_debug/bin")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG})
else()
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_BINARY_DIR}/examples)
endif()

include_directories ( ${EIGEN3_INCLUDE_DIRS} )

#==
# Build Targets
#==

function(create_executable app_name file_name)
    add_executable(${app_name} ${file_name})
    set_target_properties(${app_name} PROPERTIES MACOSX_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/mac/lib")
    if(WIN32)
        target_link_libraries(${app_name} PUBLIC raisim::raisim Ws2_32)
        target_compile_options(${app_name} PRIVATE "/MP")
    else()
        target_link_libraries(${app_name} PUBLIC raisim::raisim pthread)
    endif()
    target_include_directories(${app_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endfunction()

# benchmark examples
create_executable(anymal_speed_test src/benchmark/anymal.cpp)

# server examples
create_executable(primitives src/server/primitives.cpp)
create_executable(meshes src/server/meshes.cpp)
create_executable(laikago src/server/laikago.cpp)
create_executable(balls src/server/balls.cpp)
create_executable(atlas src/server/atlas.cpp)
create_executable(heightmap src/server/heightmap.cpp)
create_executable(robots src/server/robots.cpp)
create_executable(materials src/server/material.cpp)
create_executable(heightMapUsingPng src/server/heightMapUsingPng.cpp)
create_executable(wheeledRobot src/server/wheeledRobot.cpp)
create_executable(newtonsCradle src/server/newtonsCradle.cpp)
create_executable(visualObjects src/server/visualObjects.cpp)
create_executable(rayDemo src/server/rayDemo.cpp)
create_executable(rayDemo2 src/server/rayDemo2.cpp)
create_executable(anymals src/server/anymals.cpp)
create_executable(compound src/server/compound.cpp)
create_executable(ampmotion src/server/amp_motion.cpp)

# xml reader
create_executable(xmlRader src/xml/xmlReader.cpp)