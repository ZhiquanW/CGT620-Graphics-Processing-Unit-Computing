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
CMAKE_COMMAND = /home/zhiquan/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.7319.72/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/zhiquan/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.7319.72/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release

# Include any dependencies generated for this target.
include CMakeFiles/CUDA.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CUDA.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CUDA.dir/flags.make

CMakeFiles/CUDA.dir/imgui/imgui.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/imgui/imgui.cpp.o: ../imgui/imgui.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CUDA.dir/imgui/imgui.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/imgui/imgui.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui.cpp

CMakeFiles/CUDA.dir/imgui/imgui.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/imgui/imgui.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui.cpp > CMakeFiles/CUDA.dir/imgui/imgui.cpp.i

CMakeFiles/CUDA.dir/imgui/imgui.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/imgui/imgui.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui.cpp -o CMakeFiles/CUDA.dir/imgui/imgui.cpp.s

CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.o: ../imgui/imgui_demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_demo.cpp

CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_demo.cpp > CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.i

CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_demo.cpp -o CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.s

CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.o: ../imgui/imgui_draw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_draw.cpp

CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_draw.cpp > CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.i

CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_draw.cpp -o CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.s

CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.o: ../imgui/imgui_impl_glfw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_impl_glfw.cpp

CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_impl_glfw.cpp > CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.i

CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_impl_glfw.cpp -o CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.s

CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.o: ../imgui/imgui_impl_opengl3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_impl_opengl3.cpp

CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_impl_opengl3.cpp > CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.i

CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_impl_opengl3.cpp -o CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.s

CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.o: ../imgui/imgui_widgets.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_widgets.cpp

CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_widgets.cpp > CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.i

CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/imgui/imgui_widgets.cpp -o CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.s

CMakeFiles/CUDA.dir/src/Camera.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/Camera.cpp.o: ../src/Camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/CUDA.dir/src/Camera.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/Camera.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Camera.cpp

CMakeFiles/CUDA.dir/src/Camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/Camera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Camera.cpp > CMakeFiles/CUDA.dir/src/Camera.cpp.i

CMakeFiles/CUDA.dir/src/Camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/Camera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Camera.cpp -o CMakeFiles/CUDA.dir/src/Camera.cpp.s

CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.o: ../src/ElementBufferObject.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ElementBufferObject.cpp

CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ElementBufferObject.cpp > CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.i

CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ElementBufferObject.cpp -o CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.s

CMakeFiles/CUDA.dir/src/Mesh.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/Mesh.cpp.o: ../src/Mesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/CUDA.dir/src/Mesh.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/Mesh.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Mesh.cpp

CMakeFiles/CUDA.dir/src/Mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/Mesh.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Mesh.cpp > CMakeFiles/CUDA.dir/src/Mesh.cpp.i

CMakeFiles/CUDA.dir/src/Mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/Mesh.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Mesh.cpp -o CMakeFiles/CUDA.dir/src/Mesh.cpp.s

CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.o: ../src/ShaderProgram.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ShaderProgram.cpp

CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ShaderProgram.cpp > CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.i

CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ShaderProgram.cpp -o CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.s

CMakeFiles/CUDA.dir/src/Texture.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/Texture.cpp.o: ../src/Texture.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/CUDA.dir/src/Texture.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/Texture.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Texture.cpp

CMakeFiles/CUDA.dir/src/Texture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/Texture.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Texture.cpp > CMakeFiles/CUDA.dir/src/Texture.cpp.i

CMakeFiles/CUDA.dir/src/Texture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/Texture.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/Texture.cpp -o CMakeFiles/CUDA.dir/src/Texture.cpp.s

CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.o: ../src/VertexArrayObject.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/VertexArrayObject.cpp

CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/VertexArrayObject.cpp > CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.i

CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/VertexArrayObject.cpp -o CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.s

CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.o: ../src/VertexBufferObject.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/VertexBufferObject.cpp

CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/VertexBufferObject.cpp > CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.i

CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/VertexBufferObject.cpp -o CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.s

CMakeFiles/CUDA.dir/src/ZWEngine.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/ZWEngine.cpp.o: ../src/ZWEngine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/CUDA.dir/src/ZWEngine.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/ZWEngine.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ZWEngine.cpp

CMakeFiles/CUDA.dir/src/ZWEngine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/ZWEngine.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ZWEngine.cpp > CMakeFiles/CUDA.dir/src/ZWEngine.cpp.i

CMakeFiles/CUDA.dir/src/ZWEngine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/ZWEngine.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/ZWEngine.cpp -o CMakeFiles/CUDA.dir/src/ZWEngine.cpp.s

CMakeFiles/CUDA.dir/src/custom_func.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/custom_func.cpp.o: ../src/custom_func.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/CUDA.dir/src/custom_func.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/custom_func.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/custom_func.cpp

CMakeFiles/CUDA.dir/src/custom_func.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/custom_func.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/custom_func.cpp > CMakeFiles/CUDA.dir/src/custom_func.cpp.i

CMakeFiles/CUDA.dir/src/custom_func.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/custom_func.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/custom_func.cpp -o CMakeFiles/CUDA.dir/src/custom_func.cpp.s

CMakeFiles/CUDA.dir/src/glad.c.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/glad.c.o: ../src/glad.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building C object CMakeFiles/CUDA.dir/src/glad.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/CUDA.dir/src/glad.c.o   -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/glad.c

CMakeFiles/CUDA.dir/src/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CUDA.dir/src/glad.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/glad.c > CMakeFiles/CUDA.dir/src/glad.c.i

CMakeFiles/CUDA.dir/src/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CUDA.dir/src/glad.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/glad.c -o CMakeFiles/CUDA.dir/src/glad.c.s

CMakeFiles/CUDA.dir/src/main.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object CMakeFiles/CUDA.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/main.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/main.cpp

CMakeFiles/CUDA.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/main.cpp > CMakeFiles/CUDA.dir/src/main.cpp.i

CMakeFiles/CUDA.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/main.cpp -o CMakeFiles/CUDA.dir/src/main.cpp.s

CMakeFiles/CUDA.dir/src/stb_image.cpp.o: CMakeFiles/CUDA.dir/flags.make
CMakeFiles/CUDA.dir/src/stb_image.cpp.o: ../src/stb_image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object CMakeFiles/CUDA.dir/src/stb_image.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CUDA.dir/src/stb_image.cpp.o -c /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/stb_image.cpp

CMakeFiles/CUDA.dir/src/stb_image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CUDA.dir/src/stb_image.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/stb_image.cpp > CMakeFiles/CUDA.dir/src/stb_image.cpp.i

CMakeFiles/CUDA.dir/src/stb_image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CUDA.dir/src/stb_image.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/src/stb_image.cpp -o CMakeFiles/CUDA.dir/src/stb_image.cpp.s

# Object files for target CUDA
CUDA_OBJECTS = \
"CMakeFiles/CUDA.dir/imgui/imgui.cpp.o" \
"CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.o" \
"CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.o" \
"CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.o" \
"CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.o" \
"CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.o" \
"CMakeFiles/CUDA.dir/src/Camera.cpp.o" \
"CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.o" \
"CMakeFiles/CUDA.dir/src/Mesh.cpp.o" \
"CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.o" \
"CMakeFiles/CUDA.dir/src/Texture.cpp.o" \
"CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.o" \
"CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.o" \
"CMakeFiles/CUDA.dir/src/ZWEngine.cpp.o" \
"CMakeFiles/CUDA.dir/src/custom_func.cpp.o" \
"CMakeFiles/CUDA.dir/src/glad.c.o" \
"CMakeFiles/CUDA.dir/src/main.cpp.o" \
"CMakeFiles/CUDA.dir/src/stb_image.cpp.o"

# External object files for target CUDA
CUDA_EXTERNAL_OBJECTS =

CUDA: CMakeFiles/CUDA.dir/imgui/imgui.cpp.o
CUDA: CMakeFiles/CUDA.dir/imgui/imgui_demo.cpp.o
CUDA: CMakeFiles/CUDA.dir/imgui/imgui_draw.cpp.o
CUDA: CMakeFiles/CUDA.dir/imgui/imgui_impl_glfw.cpp.o
CUDA: CMakeFiles/CUDA.dir/imgui/imgui_impl_opengl3.cpp.o
CUDA: CMakeFiles/CUDA.dir/imgui/imgui_widgets.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/Camera.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/ElementBufferObject.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/Mesh.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/ShaderProgram.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/Texture.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/VertexArrayObject.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/VertexBufferObject.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/ZWEngine.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/custom_func.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/glad.c.o
CUDA: CMakeFiles/CUDA.dir/src/main.cpp.o
CUDA: CMakeFiles/CUDA.dir/src/stb_image.cpp.o
CUDA: CMakeFiles/CUDA.dir/build.make
CUDA: CMakeFiles/CUDA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Linking CXX executable CUDA"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CUDA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CUDA.dir/build: CUDA

.PHONY : CMakeFiles/CUDA.dir/build

CMakeFiles/CUDA.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CUDA.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CUDA.dir/clean

CMakeFiles/CUDA.dir/depend:
	cd /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5 /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5 /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release /home/zhiquan/Git-Repositories/CGT620-Graphics-Processing-Unit-Computing/lab5/cmake-build-release/CMakeFiles/CUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CUDA.dir/depend

