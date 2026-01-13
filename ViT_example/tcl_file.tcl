open_project ViT
set_top ViT_Block

add_files src/weight_Conv_0.h
add_files src/weight_Conv_0.cpp
add_files src/utils.h
add_files src/types.h
add_files src/pe_Conv_0.h
add_files src/pe_Conv_0.cpp
add_files src/parameters_Conv_0.h
add_files src/normalizations.hpp
add_files src/my_types_Conv_0.h
add_files src/my_hls_video.h
add_files src/linear_opt.h
add_files src/linear.h
add_files src/line_buffer_linear.h
add_files src/line_buffer_Conv_0.h
add_files src/line_buffer_Conv_0.cpp
add_files src/bias_Conv_0.h
add_files src/bias_Conv_0.cpp
add_files src/activations.h
add_files src/ViT_Block.hpp
add_files src/ViT_Block.cpp
add_files src/V_.h
add_files src/Q_.h
add_files src/PATCH.h
add_files src/OUT_.h
add_files src/NORM_2.h
add_files src/NORM_1.h
add_files src/NORM.h
add_files src/MultiHeadAttention.hpp
add_files src/MultiHeadAttention.cpp
add_files src/MLP2.h
add_files src/MLP1.h
add_files src/K_.h
add_files src/HEAD_.h
add_files -tb src/tb.cpp -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"

open_solution "solution4" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
source "./ViT/solution4/directives.tcl"
#csynth_design
csim_design
exit
