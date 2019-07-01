function [output, rfmod_count] = add_rfmod(net, rfmod_count, rfmod_inputs, base_name)
    % ip1 = 'op4'
    % ip2 = 'op5'
    ips = size(rfmod_inputs, 2);
    base_name = sprintf([base_name '_rfmod%d'], rfmod_count);
    mrf_inputs = cell(1,ips);

    for i=1:ips
        input = rfmod_inputs{i};
        fmaps = input.dimensions;
        if(numel(fmaps) == 3)
            fmaps = fmaps(3);
        end
        input.dimensions = fmaps;

        rcu_count = 1;
        [rcu_out, rcu_count] = add_rcu(net, rcu_count, input, base_name, input);
        [mrf_inputs{i}, ~] = add_rcu(net, rcu_count, rcu_out, base_name, input);
    end

    % Multi resolution fusion
    mrf_out = add_mrf(net, mrf_inputs, base_name);
    % Chained Resdiual Pooling
    crp_out = add_crp(net, mrf_out, base_name);

    out_rcu_count = 1;

    [output, ~] = add_rcu_out(net, out_rcu_count, crp_out, base_name, crp_out);
    rfmod_count = rfmod_count + 1;
end
