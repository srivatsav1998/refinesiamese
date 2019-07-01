function net = gen_network()
    netStruct = load('modified_baseline.mat');
    net = dagnn.DagNN.loadobj(netStruct);
    branch2_out = gen_rfnetwork_branch2(net);
    branch1_out = gen_rfnetwork_branch1(net);
    add_final_layers(net, branch1_out, branch2_out);
%     inputs = {'instance', [255 255 3 1], 'exemplar', [127 127 3 1]};
%     net.print(inputs, 'Format', 'dot');
    rename_baseline(net);
    stop_baseline_learning(net);
end


% src of cfnet present in experiment folder should also be added to the path in order for the code to run;
