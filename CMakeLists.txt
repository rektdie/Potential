cmake_minimum_required(VERSION 3.25)
project(Potential C)

# General warning flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Werror")

# Debug mode
set(CMAKE_C_FLAGS_DEBUG "-Ofast -g3 -DDEBUG")

# Release mode
set(CMAKE_C_FLAGS_RELEASE "-Ofast -march=native -ffast-math -fomit-frame-pointer -DNDEBUG")

# Source files
include_directories(src)

set(EVALFILE "${CMAKE_SOURCE_DIR}/nnue.bin" CACHE FILEPATH "NNUE file to embed with incbin")

if(NOT EXISTS "${EVALFILE}")
    message(FATAL_ERROR "NNUE file not found at ${EVALFILE}. Set EVALFILE to your nnue.bin path.")
endif()

add_executable(Potential
        src/potential.c
        src/uci.c src/uci.h src/structs.h
        src/perft.c src/perft.h
        src/move.c src/move.h
        src/bit_manipulation.c src/bit_manipulation.h
        src/mask.c src/mask.h
        src/board_constants.c src/board_constants.h
        src/magic.c src/magic.h
        src/table.c src/table.h
        src/fen.c src/fen.h
        src/history.c src/history.h
        src/incbin.h
        src/timeman.c src/timeman.h
        src/evaluation.c src/evaluation.h
        src/nnue.c src/nnue.h
        src/values.h
        src/search.c src/search.h
        src/bench.c src/bench.h
        src/utils.c src/utils.h
        src/see.c
        src/see.h)

target_compile_definitions(Potential PRIVATE EVALFILE="${EVALFILE}")
target_link_libraries(Potential m)
