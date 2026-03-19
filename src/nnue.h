#ifndef POTENTIAL_NNUE_H
#define POTENTIAL_NNUE_H

#pragma once

#include <stdbool.h>

#include "structs.h"

bool nnue_load(const char *path);
void nnue_init(const char *argv0);
bool nnue_is_loaded(void);
const char *nnue_current_path(void);
void nnue_refresh_accumulators(board *position);
void nnue_acc_add_sub(board *position, int stm, int add_square, int add_piece, int sub_square, int sub_piece);
void nnue_acc_add_sub_sub(board *position, int stm, int add_square, int add_piece,
                          int sub1_square, int sub1_piece, int sub2_square, int sub2_piece);
void nnue_acc_add_add_sub_sub(board *position, int stm,
                              int add1_square, int add1_piece, int add2_square, int add2_piece,
                              int sub1_square, int sub1_piece, int sub2_square, int sub2_piece);
int nnue_evaluate(const board *position);

#endif // POTENTIAL_NNUE_H
