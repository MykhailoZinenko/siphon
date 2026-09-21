# Hey Emacs, this is a -*- makefile -*-
#----------------------------------------------------------------------------
#
# Makefile for ChipWhisperer SimpleSerial-AES Program
#
#----------------------------------------------------------------------------
# On command line:
#
# make all = Make software.
#
# make clean = Clean out built project files.
#
# make coff = Convert ELF to AVR COFF.
#
# make extcoff = Convert ELF to AVR Extended COFF.
#
# make program = Download the hex file to the device, using avrdude.
#                Please customize the avrdude settings below first!
#
# make debug = Start either simulavr or avarice as specified for debugging, 
#              with avr-gdb or avr-insight as the front end for debugging.
#
# make filename.s = Just compile filename.c into the assembler code only.
#
# make filename.i = Create a preprocessed source file for use in submitting
#                   bug reports to the GCC project.
#
# To rebuild project do "make clean" then "make all".
#----------------------------------------------------------------------------

# Target file name (without extension). 
# This is the base name of the compiled .hex file.
TARGET = simpleserial-target

# List C source files here.
# Header files (.h) are automatically pulled in.
SRC += main.c network.c
# hal/hal.c hal/stm32f3/stm32f3_hal.c hal/stm32f3/stm32f3_hal_lowlevel.c startup_fixed.S hal/stm32f3/stm32f3_sysmem.c simpleserial/simpleserial.c
SS_VER=SS_VER_2_1
PLATFORM=CWLITEARM


# -----------------------------------------------------------------------------
#debugging:
#	gcc -o debug-target debug-source.c main.c

#Add simpleserial project to build
include simpleserial/Makefile.simpleserial

FIRMWAREPATH = .
include $(FIRMWAREPATH)/Makefile.inc



# Specify additional dependencies for the target
$(TARGET): $(OBJ) network.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Rule to build network.o from network.c
network.o: network.c network.h 
	$(CC) $(CFLAGS) -c -o $@ $<