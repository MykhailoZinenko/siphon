#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 6
#define NET_L2 1

// input quantization parameters 
static const float INPUT_SCALE = 1.0f;   // inputs are already scaled in the dataset
static const int INPUT_ZERO_POINT = 0; // we use symmetric quantization

// values for layer 1
static const int8_t L1_W[NET_L1][NET_INPUTS] = {
    { 15, 11, 7, 5, 11, 7, 4, 15, 3, 12 },
    { 1, 13, 2, 8, 8, 15, 14, 0, 14, 14 },
    { 5, 6, 6, 3, 3, 8, 10, 15, 14, 5 },
    { 3, 14, 1, 7, 9, 15, 6, 12, 14, 12 },
    { 12, 5, 15, 9, 7, 3, 1, 12, 1, 4 },
    { 1, 10, 12, 1, 4, 3, 12, 5, 9, 2 }
};

static const int32_t L1_B[NET_L1] = { -13, -10, -20, -12, 7, -19 };

static const int32_t L1_M0[NET_L1] = { 309619129, 280912867, 435608978, 670115209, 344246421, 566820946 };
static const int32_t L1_N[NET_L1] = { 4, 4, 4, 4, 4, 4 };

static const int ACT1_SCALE_INV = 0; // we are not using this parameter

// values for layer 2
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 10, 5, 1, 7, 1, 6 }
};

static const int32_t L2_B[NET_L2] = { 4 };

static const int32_t L2_M0[NET_L2] = { 1281899977 };
static const int32_t L2_N[NET_L2] = { 0 };
#endif
