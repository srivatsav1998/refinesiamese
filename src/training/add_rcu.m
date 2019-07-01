function [out, rcu_count] = add_rcu(net, rcu_count, prev_layer_name, base_name, ip)
    base_name = [base_name '_' ip.name];
    base_name = sprintf([base_name '_rcu%d'], rcu_count);
    relu_count = 1;
    conv_count = 1;
    sum_count = 1;
    bnorm_count = 1;
    input_dim = ip.dimensions;
    [relu_out, relu_count] = add_relu(net, relu_count, input_dim, prev_layer_name.name, base_name);
    [conv_out, conv_count] = add_conv(net, conv_count, relu_out.dimensions, 128, relu_out.name, base_name);
    [bnorm_out, bnorm_count] = add_bnorm(net, bnorm_count, conv_out.dimensions, conv_out.dimensions, conv_out.name, base_name);
    
    [relu_out, ~] = add_relu(net, relu_count, bnorm_out.dimensions, bnorm_out.name, base_name);
    [conv_out, ~] = add_conv(net, conv_count, relu_out.dimensions, input_dim, relu_out.name, base_name);
    [bnorm_out, ~] = add_bnorm(net, bnorm_count, conv_out.dimensions, conv_out.dimensions, conv_out.name, base_name);
    
    inps = {prev_layer_name.name, bnorm_out.name};
    [sum_out, ~] = add_sum(net, sum_count, inps, bnorm_out.dimensions, base_name);

    rcu_count = rcu_count + 1;
    out.name = sum_out.name;
    out.dimensions = sum_out.dimensions;
end

function [relu_out, relu_count] = add_relu(net, relu_count, input_dim, prev_layer_name, base_name)
    base_name = sprintf([base_name '_relu%d'], relu_count);
    block = dagnn.ReLU();
    input = prev_layer_name;
    relu_count = relu_count + 1;
    net.addLayer(base_name, block, input, base_name, {});
    relu_out.name = base_name;
    relu_out.dimensions = input_dim;
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

function [sum_out, sum_count] = add_sum(net, sum_count, inputs, input_dim, base_name)
    base_name = sprintf([base_name '_sum%d'],sum_count);
    block = My_sum_layer();
    net.addLayer(base_name, block, inputs, base_name, {});
    sum_out.name = base_name;
    sum_out.dimensions = input_dim;
    sum_count = sum_count + 1;
end
