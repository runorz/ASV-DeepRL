cmake_minimum_required(VERSION 3.8)
project(example)

set (CMAKE_CXX_STANDARD 14)

include_directories(
    asv-swarm/include
    asv-swarm/dependency/tomlc99
    frugally-deep/include
    /local/software/eigen/3.2.10/include
    FunctionalPlus/include
    json/include
)

set(source
    asv-swarm/dependency/tomlc99/toml.c
    asv-swarm/source/io.c
    asv-swarm/source/regular_wave.c
    asv-swarm/source/wave.c
    asv-swarm/source/asv.c
    main.cpp
)

set(source1
    asv-swarm/dependency/tomlc99/toml.c
    asv-swarm/source/io.c
    asv-swarm/source/regular_wave.c
    asv-swarm/source/wave.c
    asv-swarm/source/asv.c
    test.cpp
)

add_executable(training ${source})

add_executable(testing ${source1})
