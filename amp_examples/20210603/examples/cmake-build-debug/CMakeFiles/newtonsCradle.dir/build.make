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
include CMakeFiles/newtonsCradle.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/newtonsCradle.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/newtonsCradle.dir/flags.make

CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.o: CMakeFiles/newtonsCradle.dir/flags.make
CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.o: ../src/server/newtonsCradle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.o -c /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/newtonsCradle.cpp

CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/newtonsCradle.cpp > CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.i

CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/newtonsCradle.cpp -o CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.s

# Object files for target newtonsCradle
newtonsCradle_OBJECTS = \
"CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.o"

# External object files for target newtonsCradle
newtonsCradle_EXTERNAL_OBJECTS =

newtonsCradle: CMakeFiles/newtonsCradle.dir/src/server/newtonsCradle.cpp.o
newtonsCradle: CMakeFiles/newtonsCradle.dir/build.make
newtonsCradle: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisim.so
newtonsCradle: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimPng.so
newtonsCradle: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimZ.so
newtonsCradle: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimODE.so
newtonsCradle: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimMine.so
newtonsCradle: CMakeFiles/newtonsCradle.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable newtonsCradle"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/newtonsCradle.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/newtonsCradle.dir/build: newtonsCradle

.PHONY : CMakeFiles/newtonsCradle.dir/build

CMakeFiles/newtonsCradle.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/newtonsCradle.dir/cmake_clean.cmake
.PHONY : CMakeFiles/newtonsCradle.dir/clean

CMakeFiles/newtonsCradle.dir/depend:
	cd /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/skoo/Works/raisim_20210214/raisimlib/examples /home/skoo/Works/raisim_20210214/raisimlib/examples /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles/newtonsCradle.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/newtonsCradle.dir/depend

