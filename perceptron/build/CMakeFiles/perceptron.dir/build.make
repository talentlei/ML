# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cl/Code/ML/ML/perceptron

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cl/Code/ML/ML/perceptron/build

# Include any dependencies generated for this target.
include CMakeFiles/perceptron.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/perceptron.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/perceptron.dir/flags.make

CMakeFiles/perceptron.dir/src/perceptron.cpp.o: CMakeFiles/perceptron.dir/flags.make
CMakeFiles/perceptron.dir/src/perceptron.cpp.o: ../src/perceptron.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/cl/Code/ML/ML/perceptron/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/perceptron.dir/src/perceptron.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/perceptron.dir/src/perceptron.cpp.o -c /home/cl/Code/ML/ML/perceptron/src/perceptron.cpp

CMakeFiles/perceptron.dir/src/perceptron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perceptron.dir/src/perceptron.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/cl/Code/ML/ML/perceptron/src/perceptron.cpp > CMakeFiles/perceptron.dir/src/perceptron.cpp.i

CMakeFiles/perceptron.dir/src/perceptron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perceptron.dir/src/perceptron.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/cl/Code/ML/ML/perceptron/src/perceptron.cpp -o CMakeFiles/perceptron.dir/src/perceptron.cpp.s

CMakeFiles/perceptron.dir/src/perceptron.cpp.o.requires:
.PHONY : CMakeFiles/perceptron.dir/src/perceptron.cpp.o.requires

CMakeFiles/perceptron.dir/src/perceptron.cpp.o.provides: CMakeFiles/perceptron.dir/src/perceptron.cpp.o.requires
	$(MAKE) -f CMakeFiles/perceptron.dir/build.make CMakeFiles/perceptron.dir/src/perceptron.cpp.o.provides.build
.PHONY : CMakeFiles/perceptron.dir/src/perceptron.cpp.o.provides

CMakeFiles/perceptron.dir/src/perceptron.cpp.o.provides.build: CMakeFiles/perceptron.dir/src/perceptron.cpp.o

CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o: CMakeFiles/perceptron.dir/flags.make
CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o: ../src/perceptron_demo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/cl/Code/ML/ML/perceptron/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o -c /home/cl/Code/ML/ML/perceptron/src/perceptron_demo.cpp

CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/cl/Code/ML/ML/perceptron/src/perceptron_demo.cpp > CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.i

CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/cl/Code/ML/ML/perceptron/src/perceptron_demo.cpp -o CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.s

CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.requires:
.PHONY : CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.requires

CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.provides: CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.requires
	$(MAKE) -f CMakeFiles/perceptron.dir/build.make CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.provides.build
.PHONY : CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.provides

CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.provides.build: CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o

# Object files for target perceptron
perceptron_OBJECTS = \
"CMakeFiles/perceptron.dir/src/perceptron.cpp.o" \
"CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o"

# External object files for target perceptron
perceptron_EXTERNAL_OBJECTS =

perceptron: CMakeFiles/perceptron.dir/src/perceptron.cpp.o
perceptron: CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o
perceptron: CMakeFiles/perceptron.dir/build.make
perceptron: CMakeFiles/perceptron.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable perceptron"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/perceptron.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/perceptron.dir/build: perceptron
.PHONY : CMakeFiles/perceptron.dir/build

CMakeFiles/perceptron.dir/requires: CMakeFiles/perceptron.dir/src/perceptron.cpp.o.requires
CMakeFiles/perceptron.dir/requires: CMakeFiles/perceptron.dir/src/perceptron_demo.cpp.o.requires
.PHONY : CMakeFiles/perceptron.dir/requires

CMakeFiles/perceptron.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/perceptron.dir/cmake_clean.cmake
.PHONY : CMakeFiles/perceptron.dir/clean

CMakeFiles/perceptron.dir/depend:
	cd /home/cl/Code/ML/ML/perceptron/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cl/Code/ML/ML/perceptron /home/cl/Code/ML/ML/perceptron /home/cl/Code/ML/ML/perceptron/build /home/cl/Code/ML/ML/perceptron/build /home/cl/Code/ML/ML/perceptron/build/CMakeFiles/perceptron.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/perceptron.dir/depend

