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
include CMakeFiles/rayDemo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rayDemo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rayDemo.dir/flags.make

CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.o: CMakeFiles/rayDemo.dir/flags.make
CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.o: ../src/server/rayDemo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.o -c /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/rayDemo.cpp

CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/rayDemo.cpp > CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.i

CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skoo/Works/raisim_20210214/raisimlib/examples/src/server/rayDemo.cpp -o CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.s

# Object files for target rayDemo
rayDemo_OBJECTS = \
"CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.o"

# External object files for target rayDemo
rayDemo_EXTERNAL_OBJECTS =

rayDemo: CMakeFiles/rayDemo.dir/src/server/rayDemo.cpp.o
rayDemo: CMakeFiles/rayDemo.dir/build.make
rayDemo: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisim.so
rayDemo: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimPng.so
rayDemo: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimZ.so
rayDemo: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimODE.so
rayDemo: /home/skoo/Works/raisim_20210214/raisimlib/raisim/linux/lib/libraisimMine.so
rayDemo: CMakeFiles/rayDemo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rayDemo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rayDemo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rayDemo.dir/build: rayDemo

.PHONY : CMakeFiles/rayDemo.dir/build

CMakeFiles/rayDemo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rayDemo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rayDemo.dir/clean

CMakeFiles/rayDemo.dir/depend:
	cd /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/skoo/Works/raisim_20210214/raisimlib/examples /home/skoo/Works/raisim_20210214/raisimlib/examples /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug /home/skoo/Works/raisim_20210214/raisimlib/examples/cmake-build-debug/CMakeFiles/rayDemo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rayDemo.dir/depend

