module full_adder(output sum,carry, input a,b,cin);
    xor x1(sum,a,b,cin);
    assign carry = (a&b)|(a&cin)|(b&cin);
endmodule