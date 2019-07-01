function output = gen_rfnetwork_branch1(net)
    % inputs = {'op1', 'op2', 'op3', 'op4', 'op5'};
    %                                  ---rf1--
    %                           ---rf2----
    %                    ----rf3----
    %             ----rf4----
    % inputs is a struct of struct where each struct contains information like input name and input dimensions
    % ip1 = input{1}
    % ip1.name = 'op1'
    % ip1.dim = [61 61 96];
    base_name = 'branch1';
    inputs{1}.name = 'br1_x4';
    inputs{1}.fmaps = 96;
    inputs{1}.dimensions = [29 29 96];

    inputs{2}.name = 'br1_x8';
    inputs{2}.fmaps = 256;
    inputs{2}.dimensions = [23 23 256];

    inputs{3}.name = 'br1_x11';
    inputs{3}.fmaps = 384;
    inputs{3}.dimensions = [21 21 384];

    inputs{4}.name = 'br1_x14';
    inputs{4}.fmaps = 384;
    inputs{4}.dimensions = [19 19 384];

    inputs{5}.name = 'br1_out';
    inputs{5}.fmaps = 32;
    inputs{5}.dimensions = [17 17 32];

    % net = dagnn.DagNN();
    rfmod_count = 1;
    inps = {inputs{1}, inputs{2}, inputs{3}, inputs{4}, inputs{5}};
    [out_rf1, rfmod_count] = add_rfmod(net, rfmod_count, inps, base_name);
    
%     inps = {inputs{3}, inputs{4}, inputs{5}};
%     [out_rf1, rfmod_count] = add_rfmod(net, rfmod_count, inps, base_name);
%     inps = {inputs{1}, inputs{2}, out_rf1};
%     [out_rf2, rfmod_count] = add_rfmod(net, rfmod_count, inps, base_name);
%     inps = {inputs{3}, out_rf1};
%     [out_rf2, rfmod_count] = add_rfmod(net, rfmod_count, inps, base_name);
%     inps = {inputs{2}, out_rf2};
% 	  [out_rf3, rfmod_count] = add_rfmod(net, rfmod_count, inps, base_name);
%     inps = {inputs{1}, out_rf3};
%     [out_rf4, rfmod_count] = add_rfmod(net, rfmod_count, inps, base_name);

    % test_inp = {inputs{1}.name, inputs{1}.dimensions, inputs{2}.name, inputs{2}.dimensions, inputs{3}.name, inputs{3}.dimensions, inputs{4}.name, inputs{4}.dimensions, inputs{5}.name, inputs{5}.dimensions};
    % net.print(test_inp, 'Format', 'dot');
    %
    % net.initParams();
    % rand_inp1 = randn(61, 61, 96, 'single');
    % rand_inp2 = randn(55, 55, 256, 'single');
    % rand_inp3 = randn(53, 53, 384, 'single');
    % rand_inp4 = randn(51, 51, 384, 'single');
    % rand_inp5 = randn(49, 49, 32, 'single');
    % eval_inp = {inputs{1}.name, rand_inp1, inputs{2}.name, rand_inp2, inputs{3}.name, rand_inp3, inputs{4}.name, rand_inp4, inputs{5}.name, rand_inp5};
    % net.eval(eval_inp);
    output = out_rf1;
end
