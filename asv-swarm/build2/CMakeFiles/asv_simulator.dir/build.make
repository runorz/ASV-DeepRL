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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.17.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.17.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zhangrun/asv-swarm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zhangrun/asv-swarm/build2

# Include any dependencies generated for this target.
include CMakeFiles/asv_simulator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/asv_simulator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/asv_simulator.dir/flags.make

CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.o: CMakeFiles/asv_simulator.dir/flags.make
CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.o: ../dependency/tomlc99/toml.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhangrun/asv-swarm/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.o   -c /Users/zhangrun/asv-swarm/dependency/tomlc99/toml.c

CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/zhangrun/asv-swarm/dependency/tomlc99/toml.c > CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.i

CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/zhangrun/asv-swarm/dependency/tomlc99/toml.c -o CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.s

CMakeFiles/asv_simulator.dir/source/io.c.o: CMakeFiles/asv_simulator.dir/flags.make
CMakeFiles/asv_simulator.dir/source/io.c.o: ../source/io.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhangrun/asv-swarm/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/asv_simulator.dir/source/io.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/asv_simulator.dir/source/io.c.o   -c /Users/zhangrun/asv-swarm/source/io.c

CMakeFiles/asv_simulator.dir/source/io.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/asv_simulator.dir/source/io.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/zhangrun/asv-swarm/source/io.c > CMakeFiles/asv_simulator.dir/source/io.c.i

CMakeFiles/asv_simulator.dir/source/io.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/asv_simulator.dir/source/io.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/zhangrun/asv-swarm/source/io.c -o CMakeFiles/asv_simulator.dir/source/io.c.s

CMakeFiles/asv_simulator.dir/source/regular_wave.c.o: CMakeFiles/asv_simulator.dir/flags.make
CMakeFiles/asv_simulator.dir/source/regular_wave.c.o: ../source/regular_wave.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhangrun/asv-swarm/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/asv_simulator.dir/source/regular_wave.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/asv_simulator.dir/source/regular_wave.c.o   -c /Users/zhangrun/asv-swarm/source/regular_wave.c

CMakeFiles/asv_simulator.dir/source/regular_wave.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/asv_simulator.dir/source/regular_wave.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/zhangrun/asv-swarm/source/regular_wave.c > CMakeFiles/asv_simulator.dir/source/regular_wave.c.i

CMakeFiles/asv_simulator.dir/source/regular_wave.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/asv_simulator.dir/source/regular_wave.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/zhangrun/asv-swarm/source/regular_wave.c -o CMakeFiles/asv_simulator.dir/source/regular_wave.c.s

CMakeFiles/asv_simulator.dir/source/wave.c.o: CMakeFiles/asv_simulator.dir/flags.make
CMakeFiles/asv_simulator.dir/source/wave.c.o: ../source/wave.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhangrun/asv-swarm/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/asv_simulator.dir/source/wave.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/asv_simulator.dir/source/wave.c.o   -c /Users/zhangrun/asv-swarm/source/wave.c

CMakeFiles/asv_simulator.dir/source/wave.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/asv_simulator.dir/source/wave.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/zhangrun/asv-swarm/source/wave.c > CMakeFiles/asv_simulator.dir/source/wave.c.i

CMakeFiles/asv_simulator.dir/source/wave.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/asv_simulator.dir/source/wave.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/zhangrun/asv-swarm/source/wave.c -o CMakeFiles/asv_simulator.dir/source/wave.c.s

CMakeFiles/asv_simulator.dir/source/asv.c.o: CMakeFiles/asv_simulator.dir/flags.make
CMakeFiles/asv_simulator.dir/source/asv.c.o: ../source/asv.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhangrun/asv-swarm/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/asv_simulator.dir/source/asv.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/asv_simulator.dir/source/asv.c.o   -c /Users/zhangrun/asv-swarm/source/asv.c

CMakeFiles/asv_simulator.dir/source/asv.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/asv_simulator.dir/source/asv.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/zhangrun/asv-swarm/source/asv.c > CMakeFiles/asv_simulator.dir/source/asv.c.i

CMakeFiles/asv_simulator.dir/source/asv.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/asv_simulator.dir/source/asv.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/zhangrun/asv-swarm/source/asv.c -o CMakeFiles/asv_simulator.dir/source/asv.c.s

CMakeFiles/asv_simulator.dir/source/pid_controller.c.o: CMakeFiles/asv_simulator.dir/flags.make
CMakeFiles/asv_simulator.dir/source/pid_controller.c.o: ../source/pid_controller.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zhangrun/asv-swarm/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/asv_simulator.dir/source/pid_controller.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/asv_simulator.dir/source/pid_controller.c.o   -c /Users/zhangrun/asv-swarm/source/pid_controller.c

CMakeFiles/asv_simulator.dir/source/pid_controller.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/asv_simulator.dir/source/pid_controller.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/zhangrun/asv-swarm/source/pid_controller.c > CMakeFiles/asv_simulator.dir/source/pid_controller.c.i

CMakeFiles/asv_simulator.dir/source/pid_controller.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/asv_simulator.dir/source/pid_controller.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/zhangrun/asv-swarm/source/pid_controller.c -o CMakeFiles/asv_simulator.dir/source/pid_controller.c.s

# Object files for target asv_simulator
asv_simulator_OBJECTS = \
"CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.o" \
"CMakeFiles/asv_simulator.dir/source/io.c.o" \
"CMakeFiles/asv_simulator.dir/source/regular_wave.c.o" \
"CMakeFiles/asv_simulator.dir/source/wave.c.o" \
"CMakeFiles/asv_simulator.dir/source/asv.c.o" \
"CMakeFiles/asv_simulator.dir/source/pid_controller.c.o"

# External object files for target asv_simulator
asv_simulator_EXTERNAL_OBJECTS =

libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/dependency/tomlc99/toml.c.o
libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/source/io.c.o
libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/source/regular_wave.c.o
libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/source/wave.c.o
libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/source/asv.c.o
libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/source/pid_controller.c.o
libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/build.make
libasv_simulator.dylib: CMakeFiles/asv_simulator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zhangrun/asv-swarm/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking C shared library libasv_simulator.dylib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/asv_simulator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/asv_simulator.dir/build: libasv_simulator.dylib

.PHONY : CMakeFiles/asv_simulator.dir/build

CMakeFiles/asv_simulator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/asv_simulator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/asv_simulator.dir/clean

CMakeFiles/asv_simulator.dir/depend:
	cd /Users/zhangrun/asv-swarm/build2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zhangrun/asv-swarm /Users/zhangrun/asv-swarm /Users/zhangrun/asv-swarm/build2 /Users/zhangrun/asv-swarm/build2 /Users/zhangrun/asv-swarm/build2/CMakeFiles/asv_simulator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/asv_simulator.dir/depend
