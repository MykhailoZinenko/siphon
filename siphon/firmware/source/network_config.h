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
    { -6, -3, 7, -4, 6, 1, 7, 6, -7, -1 },
    { 1, 4, 3, 1, 2, 0, -7, 2, 5, 7 },
    { -1, 6, -7, 3, 5, -6, -3, 6, 2, 3 },
    { -3, -2, -1, -4, 7, 6, 3, 5, -6, -2 },
    { 3, 7, -3, 3, 0, -7, 0, -2, -6, -5 },
    { -3, -2, -4, 7, 7, 1, 0, -2, 1, -5 }
};

static const int32_t L1_B[NET_L1] = { 4, -9, 2, 9, -16, 17 };

static const int32_t L1_M0[NET_L1] = { 617572004, 675389917, 1066828149, 1156238072, 931036666, 1203884375 };
static const int32_t L1_N[NET_L1] = { 4, 4, 4, 4, 4, 4 };

static const int ACT1_SCALE_INV = 0; // we are not using this parameter

// values for layer 2
static const int8_t L2_W[NET_L2][NET_L1] = {
    { 0, -1, -6, -5, -1, 0 }
};

static const int32_t L2_B[NET_L2] = { -3 };

static const int32_t L2_M0[NET_L2] = { 1281899977 };
static const int32_t L2_N[NET_L2] = { 0 };
#endif
