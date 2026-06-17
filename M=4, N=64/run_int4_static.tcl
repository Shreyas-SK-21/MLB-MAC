set DESIGN_NAME "int_mac_M4_N64"
set TECH_LEF "NangateOpenCellLibrary.tech.lef"
set MACRO_LEF "NangateOpenCellLibrary.macro.lef"
set LIB_FILE "NangateOpenCellLibrary_typical.lib"
# Ensure your synthesized 4-bit INT netlist is named correctly here
set NETLIST_FILE "synth_int_mac_M4_N64.v" 

read_lef $TECH_LEF
read_lef $MACRO_LEF
read_liberty $LIB_FILE
read_verilog $NETLIST_FILE
link_design $DESIGN_NAME

create_clock -name clk -period 10.0 [get_ports clk]
set_input_delay  -clock clk 1.0 [all_inputs]
set_output_delay -clock clk 1.0 [all_outputs]

# Force theoretical 50% toggle rate
set_power_activity -global -activity 0.5

puts "\n========================================"
puts " AREA REPORT: $DESIGN_NAME (STATIC)"
puts "========================================"
report_design_area

puts "\n========================================"
puts " POWER REPORT: $DESIGN_NAME (STATIC)"
puts "========================================"
report_power
exit