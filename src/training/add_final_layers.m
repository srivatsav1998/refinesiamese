function output = add_final_layers(net, branch1_out, branch2_out)
    inps = {branch1_out.name, branch2_out.name};
    net.addLayer('join_xcorr', XCorr(), inps, {'join_out'}, {});
%     block = dagnn.Conv();
%     block.size = [1 1 1 1];
%     block.hasBias = true;
%     block.pad = 0;
%     block.stride = 1;
%     params = struct('name', {}, 'value', {}, 'learningRate', [], 'weightDecay', []);
%     params(1).name = 'convFilters';
%     params(1).value = single(1);
%     params(1).learningRate = 0;
%     params(1).weightDecay = 1;
%     params(2).name = 'convBias';
%     params(2).value = single(0);
%     params(2).learningRate = 1;
%     params(2).weightDecay = 0;
%     
%     net.addLayer('out_conv', block, {'join_xcorr'}, {'score'}, {params.name});
%     for p = 1:numel(params)
%         pindex = net.getParamIndex(params(p).name);
%         if ~isempty(params(p).value)
%             net.params(pindex).value = params(p).value;
%         end
%         if ~isempty(params(p).learningRate)
%             net.params(pindex).learningRate = params(p).learningRate;
%         end
%         if ~isempty(params(p).weightDecay)
%             net.params(pindex).weightDecay = params(p).weightDecay;
%         end
%     end
    add_bnorm_final(net, 1, 1, 1, {'join_out'});
end

function [bnorm_out, bnorm_count] = add_bnorm_final(net, bnorm_count, input_dim, output_dim, prev_layer_name)
    base_name = ['final_adj'];
    params = struct(...
            'name', {}, ...
            'value', {}, ...
            'learningRate', [], ...
            'weightDecay', []);
    params(1).name = [base_name '_m'];
    params(1).value = single(ones(input_dim,1));
    params(1).learningRate = 2;
    params(1).weightDecay = 0;
    
    params(2).name = [base_name '_b'];
    params(2).value = single(zeros(input_dim, 1));
    params(2).learningRate = 1;
    params(2).weightDecay = 0;
    
    params(3).name = [base_name '_x'];
    params(3).value = single(zeros(input_dim,2));
    params(3).learningRate = 0.05;
    params(3).weightDecay = 0;
    
    block = dagnn.BatchNorm();
    
    net.addLayer(base_name, block, prev_layer_name, {'score'}, {params.name});
    for p = 1:numel(params)
        pindex = net.getParamIndex(params(p).name);
        if ~isempty(params(p).value)
            net.params(pindex).value = params(p).value;
        end
        if ~isempty(params(p).learningRate)
            net.params(pindex).learningRate = params(p).learningRate;
        end
        if ~isempty(params(p).weightDecay)
            net.params(pindex).weightDecay = params(p).weightDecay;
        end
    end
    
    bnorm_out.name = base_name;
    bnorm_out.dimensions = output_dim;
    bnorm_count = bnorm_count + 1;
end
