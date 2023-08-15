############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
list_part -name u280

open_project tfm_v2
set_top tfm


list_part
add_files tfm_top.cpp

#add_files -tb main.cc -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xcku19p-ffvb2104-1-i}
#set_part 
#xc6vlx240tff1156-1
#-board u200
#{xczu7ev-2ffvc1156-1}
#{xc7z045-ffg900-2}
create_clock -period 5 -name default
config_compile -no_signed_zeros=0 -unsafe_math_optimizations=0
config_sdx -target none
config_export -format ip_catalog -rtl verilog -vivado_optimization_level 2 -vivado_phys_opt place -vivado_report_level 0
config_schedule -effort medium -enable_dsp_full_reg=0 -relax_ii_for_timing=0 -verbose=0
config_bind -effort medium
set_clock_uncertainty 12.5%
#source "./solution1/directives.tcl"
#csim_design
csynth_design
