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
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, -2, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 2, 0, 0, 0, 0, 0, 0, 0, 0 },
    { -2, -2, 0, 0, 0, 0, 0, 0, 0, 0 },
    { -2, 2, 0, 0, 0, 0, 0, 0, 0, 0 }
};

static const int32_t L1_B[NET_L1] = { -9, 0, -61, -63, -63, -59 };

static const int32_t L1_M0[NET_L1] = { 1759218604, 1462571216, 1499413719, 1270239266, 1296455441, 2056387487 };
static const int32_t L1_N[NET_L1] = { 13, 2, 4, 4, 4, 5 };

static const int ACT1_SCALE_INV = 0; // we are not using this parameter

// values for layer 2
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 4, -1, 7, -3, 6, -2 }
};

static const int32_t L2_B[NET_L2] = { 0 };

static const int32_t L2_M0[NET_L2] = { 1281899977 };
static const int32_t L2_N[NET_L2] = { 0 };
#endif
