module fp8_mlb_top(output signed [23:0] wide_integer_sum,output signed [8:0] shared_exponent,output mac_done,input clk,rst,valid_in,input [4095:0] fp8_activations,fp8_weights);
    wire [2047:0] axi_planes;
    wire [2047:0] awi_planes;
    wire [3:0] max_exp_x;
    wire [3:0] max_exp_w;
    bfp_aligner align_x (.clk(clk),.fp8_vec(fp8_activations),.aligned_planes(axi_planes),.max_exp(max_exp_x));
    bfp_aligner align_w (.clk(clk),.fp8_vec(fp8_weights),.aligned_planes(awi_planes),.max_exp(max_exp_w));
    // ---------------- Sign handling ----------------
    // sign_k is derived directly from the raw (un-registered) inputs, but
    // axi_planes/awi_planes now arrive ONE cycle late from the pipelined
    // aligner above. Register sign_k by 1 cycle so pos_mask/neg_mask line
    // up in time with the aligned planes they are ANDed against.
    wire [511:0] sign_k;
    genvar i;
    generate
        for (i=0;i<512;i=i+1) begin
            assign sign_k[i]=fp8_activations[i*8+7]^fp8_weights[i*8+7];
        end
    endgenerate
    reg [511:0] sign_k_d1;
    always @(posedge clk) sign_k_d1 <= sign_k;
    wire [511:0] pos_mask = ~sign_k_d1;
    wire [511:0] neg_mask = sign_k_d1;
    wire [2047:0] axi_planes_pos;
    wire [2047:0] axi_planes_neg;
    assign axi_planes_pos[511:0]    = axi_planes[511:0]    & pos_mask;
    assign axi_planes_pos[1023:512]  = axi_planes[1023:512]  & pos_mask;
    assign axi_planes_pos[1535:1024] = axi_planes[1535:1024] & pos_mask;
    assign axi_planes_pos[2047:1536] = axi_planes[2047:1536] & pos_mask;
    
    assign axi_planes_neg[511:0]    = axi_planes[511:0]    & neg_mask;
    assign axi_planes_neg[1023:512]  = axi_planes[1023:512]  & neg_mask;
    assign axi_planes_neg[1535:1024] = axi_planes[1535:1024] & neg_mask;
    assign axi_planes_neg[2047:1536] = axi_planes[2047:1536] & neg_mask;

    // ---------------- FSM: Input Time-Multiplexing ----------------
    // valid_in is delayed by 1 cycle to match the aligner's registered
    // latency -- otherwise the FSM would fire before axi_planes/awi_planes
    // (and sign_k_d1) are actually valid.
    reg valid_in_d1;
    always @(posedge clk) begin
        if (rst) valid_in_d1 <= 1'b0;
        else     valid_in_d1 <= valid_in;
    end

    reg is_neg_phase;
    reg internal_valid;
    reg [1:0] state;
    always @(posedge clk) begin
        if (rst) begin
            state <= 2'd0;
            internal_valid <= 1'b0;
            is_neg_phase <= 1'b0;
        end else begin
            case (state)
                2'd0: begin
                    if (valid_in_d1) begin
                        internal_valid <= 1'b1;
                        is_neg_phase <= 1'b0; // Launch Positive mask first
                        state <= 2'd1;
                    end else begin
                        internal_valid <= 1'b0;
                    end
                end
                2'd1: begin
                    internal_valid <= 1'b1;
                    is_neg_phase <= 1'b1; // Immediately launch Negative mask next clock
                    state <= 2'd2;
                end
                2'd2: begin
                    internal_valid <= 1'b0;
                    state <= 2'd0; // Wait for next valid_in from testbench
                end
            endcase
        end
    end

    // Input MUX: Select positive or negative planes based on the FSM phase
    wire [2047:0] muxed_axi_planes = is_neg_phase ? axi_planes_neg : axi_planes_pos;

    // ---------------- SINGLE MLB_4 ARRAY (Resource Shared) ----------------
    wire signed [22:0] shared_sum_out;
    wire shared_done;

    MLB_4 mac_shared (.mlb(shared_sum_out),.done(shared_done),.axi(muxed_axi_planes),.awi(awi_planes),.clk(clk),.rst(rst),.valid_in(internal_valid));

    // ---------------- Output FSM: Catch & Combine ----------------
    reg signed [22:0] pos_reg;
    reg got_pos;
    reg signed [23:0] result_reg;
    reg done_reg;

    always @(posedge clk) begin
        if (rst) begin
            pos_reg <= 23'sd0;
            got_pos <= 1'b0;
            result_reg <= 24'sd0;
            done_reg <= 1'b0;
        end else begin
            done_reg <= 1'b0;
            
            if (shared_done) begin
                if (!got_pos) begin
                    // First pulse out of the pipeline is the Positive Sum
                    pos_reg <= shared_sum_out;
                    got_pos <= 1'b1;
                end else begin
                    // Second pulse out of the pipeline is the Negative Sum
                    result_reg <= pos_reg - shared_sum_out; // pos_sum - neg_sum
                    done_reg <= 1'b1; // Fire final done signal to testbench
                    got_pos <= 1'b0;  // Reset for the next inference
                end
            end
        end
    end

    assign wide_integer_sum = result_reg;
    assign mac_done         = done_reg;
    assign shared_exponent  = $signed({5'b0, max_exp_x}) + $signed({5'b0, max_exp_w}) - 9'sd20; // 14 (2x E4M3 bias) + 6 (2x hidden-bit integer scale 2^3)

endmodule
