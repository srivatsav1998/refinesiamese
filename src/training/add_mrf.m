function mrf_out = add_mrf(net, mrf_inputs, base_name)
    base_name = [base_name '_mrf'];
    % input1 should pass through a conv layer
    % input2 should pass through a conv layer
    % these two ouputs are fused together

    % input1 convolution
    ips = size(mrf_inputs,2);
    sum_inps = cell(1, ips);
    sum_dim = 0;
    for i=1:ips
        ip_base = sprintf([base_name '_inp%d'],i);
        [conv_out, ~] = add_conv(net, 1, mrf_inputs{i}.dimensions, 64, mrf_inputs{i}.name, ip_base);
        [bnorm_out, ~] = add_bnorm(net, 1, conv_out.dimensions, conv_out.dimensions, conv_out.name, ip_base);
        sum_inps{i} = bnorm_out.name;
        sum_dim = bnorm_out.dimensions;
    end

    % target_size = max(inp1_hw, inp2_hw);
    % resized_inp1 = imresize(input1, target_size, 'bicubic');
    % resized_inp2 = imresize(input2, target_size, 'bicubic');
    sum_count = 1;

    [sum_out, ~] = add_sum(net, sum_count, sum_inps, sum_dim, base_name);
    mrf_out = sum_out;

end

function [conv_out, conv_count] = add_conv(net, conv_count, input_dim, output_dim, prev_layer_name, base_name)
    base_name = sprintf([base_name '_conv%d'], conv_count);
    params = struct(...
                'name', {}, ...
                'value', {}, ...
                'learningRate', [], ...
                'weightDecay', []);
    sc = sqrt(2/(3*3*output_dim)) ;
    filters = sc * randn(3, 3, input_dim, output_dim, 'single');
    sz = size(filters);
    params(1).name = [base_name '_filter'];
    params(1).value = filters;
    params(1).learningRate = 1;
    params(1).weightDecay = 1;
    biases = zeros(1, output_dim, 'single');
    params(2).name = [base_name '_bias'];
    params(2).value = biases;
    params(2).learningRate = 1;
    params(2).weightDecay = 1;

    pad_size = 1;
    block = dagnn.Conv();
    block.size = sz;
    block.hasBias = true;
    block.pad = pad_size;
    block.stride = 1;

    net.addLayer(base_name, block, prev_layer_name, base_name, {params.name});
    for p = 1:numel(params)
        pindex = net.getParamIndex(params(p).name) ;
        if ~isempty(params(p).value)
            net.params(pindex).value = params(p).value ;
        end
        if ~isempty(params(p).learningRate)
            net.params(pindex).learningRate = params(p).learningRate ;
        end
        if ~isempty(params(p).weightDecay)
            net.params(pindex).weightDecay = params(p).weightDecay ;
        end
    end

    conv_out.name = base_name;
    conv_out.dimensions = output_dim;
    conv_count = conv_count + 1;
end

function [sum_out, sum_count] = add_sum(net, sum_count, inputs, input_dim, base_name)
    base_name = sprintf([base_name '_sum%d'],sum_count);
    block = My_sum_layer();
    net.addLayer(base_name, block, inputs, base_name, {});
    sum_out.name = base_name;
    sum_out.dimensions = input_dim;
    sum_count = sum_count + 1;
end

function [bnorm_out, bnorm_count] = add_bnorm(net, bnorm_count, input_dim, output_dim, prev_layer_name, base_name)
    base_name = sprintf([base_name '_bnorm%d'], bnorm_count);
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

    net.addLayer(base_name, block, prev_layer_name, base_name, {params.name});
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
