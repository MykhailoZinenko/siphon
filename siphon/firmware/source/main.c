#include "hal.h"
#include "simpleserial.h"
#include "network.h"          
#include "network_config.h"   

// a bigger sled for better visual separation
#define NOP_SLED_50() do { \
    for(int i=0; i<200; i++) { __asm__ __volatile__ ("nop"); } \
} while(0)

uint8_t debug_handle(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    // sanity check
    // 'd' command expects 10 bytes, returns the same 10 bytes
    simpleserial_put('d', len, buf);
    return 0;
}


uint8_t handle(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    trigger_high();
    
    NOP_SLED_50(); 

    int32_t result = network_forward((int8_t*)buf);
    
    NOP_SLED_50(); 

    trigger_low();
    simpleserial_put('r', sizeof(int32_t), (uint8_t*)&result);
    return 0;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();
    simpleserial_addcmd('p', 10, handle);
    simpleserial_addcmd('d', 10, debug_handle);
    while(1) simpleserial_get();
}