#include "network_config.h"
#include <stdint.h>

/*
110100010000001
010100010000010
100000000000011
100100010000100
010000000000101
110000000000110
000100010000111
000100010001000
110000000001001
010000000001010
100100010001011
100000000001100
010100010001101
110100010001110

*/


#define NOP_SLED_100() do { \
    for(int i=0; i<100; i++) { __asm__ __volatile__ ("nop"); } \
} while(0)


#define NOP_SLED_50() do { \
    for(int i=0; i<30; i++) { __asm__ __volatile__ ("nop"); } \
} while(0)


int32_t network_forward(const int8_t input[])
{
    // layer 1 input is the function argument 'input'
    // layer 1 output is stored in hidden_q
    int8_t hidden_q[NET_L1];
    
    // pointer to current input for the active layer loop
    const int8_t* current_input = input;
    int32_t current_input_size = NET_INPUTS;
    
    for (int layer = 0; layer < 2; layer++) {
        // NOP_SLED_100();
        for(int n=0; n<20; n++) { __asm__ __volatile__ ("nop"); }
        int num_neurons = (layer == 0) ? NET_L1 : 1;
        
        for (int o = 0; o < num_neurons; o++) {
            int32_t acc = 0;
            
            for (int i = 0; i < current_input_size; i++) {
                int8_t w = (layer == 0) ? L1_W[o][i] : L2_W[o][i];
                acc += (int32_t)current_input[i] * w;
            }

            // add bias and scale
            int32_t bias = (layer == 0) ? L1_B[o] : L2_B[o];
            int32_t m0   = (layer == 0) ? L1_M0[o] : L2_M0[o];
            int32_t n    = (layer == 0) ? L1_N[o] : L2_N[o];
            
            acc += bias;

            // fixed-point multiplication with rounding
            int64_t temp = (int64_t)acc * m0;
            temp += (1LL << (31 + n - 1));
            int32_t activation = (int32_t)(temp >> (31 + n));

            if (layer == 0) {
                // clamp to 3-bit range [0, 7]
                if (activation < 0) activation = 0;
                if (activation > 7) activation = 7;
                hidden_q[o] = (int8_t)activation;
            } else {
                // layer 2 is the final logit, return immediately
                return activation;
            }
            // NOP_SLED_100();
        }
        
        current_input = hidden_q;
        current_input_size = NET_L1;
    }

    return 0; // should not reach here
}