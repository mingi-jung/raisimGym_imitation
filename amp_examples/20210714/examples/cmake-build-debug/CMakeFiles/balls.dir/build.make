# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/skoo/Works/clion-2020.3.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/skoo/Works/clion-2020.3.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/skoo/Works/raisim_20210214/raisimlib/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/balls.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/balls.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/balls.dir/flags.make

CMakeFiles/balls.dir/src/server/balls.cpp.o: CMakeFiles/balls.dir/flags.make
CMakeFiles/balls.dir/src/server/balls.cpp.o: ../src/server/balls.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/balls.dir/src/server/balls.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/balls.dir/src/server/balls.cpp.o -c /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/balls.cpp

CMakeFiles/balls.dir/src/server/balls.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/balls.dir/src/server/balls.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/balls.cpp > CMakeFiles/balls.dir/src/server/balls.cpp.i

CMakeFiles/balls.dir/src/server/balls.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/balls.dir/src/server/balls.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/balls.cpp -o CMakeFiles/balls.dir/src/server/balls.cpp.s

# Object files for target balls
balls_OBJECTS = \
"CMakeFiles/balls.dir/src/server/balls.cpp.o"

# External object files for target balls
balls_EXTERNAL_OBJECTS =

balls: CMakeFiles/balls.dir/src/server/balls.cpp.o
balls: CMakeFiles/balls.dir/build.make
balls: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisim.so
balls: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimPng.so
balls: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimZ.so
balls: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimODE.so
balls: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimMine.so
balls: CMakeFiles/balls.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable balls"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/balls.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/balls.dir/build: balls

.PHONY : CMakeFiles/balls.dir/build

CMakeFiles/balls.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/balls.dir/cmake_clean.cmake
.PHONY : CMakeFiles/balls.dir/clean

CMakeFiles/balls.dir/depend:
	cd /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/skoo/Works/raisim_20210214/raisimlib/examples /home/skoo/Works/raisim_20210214/raisimlib/examples /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles/balls.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/balls.dir/depend

