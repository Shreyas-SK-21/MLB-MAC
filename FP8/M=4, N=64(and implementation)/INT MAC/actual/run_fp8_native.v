# Capture start time to measure execution time
set start_time [clock milliseconds]

set DESIGN_NAME "FP8_NATIVE_MAC_64" 
set TECH_LEF "NangateOpenCellLibrary.tech.lef"
set MACRO_LEF "NangateOpenCellLibrary.macro.lef"
set LIB_FILE "NangateOpenCellLibrary_typical.lib"
set NETLIST_FILE "fp8_native_synth.v" 

# Load standard cell libraries and synthesized netlist
read_lef $TECH_LEF
read_lef $MACRO_LEF
read_liberty $LIB_FILE
read_verilog $NETLIST_FILE
link_design $DESIGN_NAME

# Apply timing constraints
create_clock -name clk -period 10.0 [get_ports clk]
set_input_delay  -clock clk 1.0 [all_inputs]
set_output_delay -clock clk 1.0 [all_outputs]

# Force theoretical 50% toggle rate across all nets to estimate dynamic power
set_power_activity -global -activity 0.5

puts "\n========================================"
puts " AREA REPORT: $DESIGN_NAME (STATIC)"
puts "========================================"
report_design_area

puts "\n========================================"
puts " POWER REPORT: $DESIGN_NAME (STATIC)"
puts "========================================"
report_power

puts "\n========================================"
puts " TIMING & LATENCY REPORT: $DESIGN_NAME"
puts "========================================"
# report_checks outputs the worst-case timing path. 
# The "data arrival time" at the endpoint represents your circuit's latency.
report_checks -path_delay max -format full_clock_expanded

puts "\n========================================"
puts " EXECUTION TIME"
puts "========================================"
set end_time [clock milliseconds]
set run_time_ms [expr {$end_time - $start_time}]
set run_time_s [expr {$run_time_ms / 1000.0}]
puts "Tool Execution Time: $run_time_s seconds ($run_time_ms ms)"

exit