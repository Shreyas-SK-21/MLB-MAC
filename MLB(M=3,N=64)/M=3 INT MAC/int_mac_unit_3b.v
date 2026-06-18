// int_mac_unit_3b.v
// Signed 3-bit × signed 3-bit → signed 6-bit product
// Paper: Stage 1 MAC unit element (M=3)
module int_mac_unit_3b (
    input  signed [2:0] a,
    input  signed [2:0] b,
    output signed [5:0] product
);
    assign product = a * b;
endmodule
