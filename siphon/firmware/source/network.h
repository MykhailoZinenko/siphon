#ifndef NETWORK_H
#define NETWORK_H
#include <stdint.h>

// pure integer inferene. returns fixed-point logit
int32_t network_forward(const int8_t input[]);


#endif