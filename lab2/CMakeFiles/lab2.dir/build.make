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
CMAKE_COMMAND = /home/zhiquan/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.6948.80/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/zhiquan/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.6948.80/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2

# Include any dependencies generated for this target.
include CMakeFiles/lab2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lab2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab2.dir/flags.make

CMakeFiles/lab2.dir/main.cu.o: CMakeFiles/lab2.dir/flags.make
CMakeFiles/lab2.dir/main.cu.o: main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/lab2.dir/main.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2/main.cu -o CMakeFiles/lab2.dir/main.cu.o

CMakeFiles/lab2.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/lab2.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/lab2.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/lab2.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target lab2
lab2_OBJECTS = \
"CMakeFiles/lab2.dir/main.cu.o"

# External object files for target lab2
lab2_EXTERNAL_OBJECTS =

CMakeFiles/lab2.dir/cmake_device_link.o: CMakeFiles/lab2.dir/main.cu.o
CMakeFiles/lab2.dir/cmake_device_link.o: CMakeFiles/lab2.dir/build.make
CMakeFiles/lab2.dir/cmake_device_link.o: CMakeFiles/lab2.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/lab2.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab2.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab2.dir/build: CMakeFiles/lab2.dir/cmake_device_link.o

.PHONY : CMakeFiles/lab2.dir/build

# Object files for target lab2
lab2_OBJECTS = \
"CMakeFiles/lab2.dir/main.cu.o"

# External object files for target lab2
lab2_EXTERNAL_OBJECTS =

lab2: CMakeFiles/lab2.dir/main.cu.o
lab2: CMakeFiles/lab2.dir/build.make
lab2: CMakeFiles/lab2.dir/cmake_device_link.o
lab2: CMakeFiles/lab2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable lab2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab2.dir/build: lab2

.PHONY : CMakeFiles/lab2.dir/build

CMakeFiles/lab2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab2.dir/clean

CMakeFiles/lab2.dir/depend:
	cd /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2 /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2 /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2 /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2 /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab2/CMakeFiles/lab2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab2.dir/depend

