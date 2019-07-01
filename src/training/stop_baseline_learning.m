function stop_baseline_learning(net)
   nparams = numel(net.params);
   for i=1:nparams
       param_name = net.params(i).name;
       parts = strsplit(param_name, '_');
       if strcmp(parts{1}, 'br')
           net.params(i).learningRate = 0;
           net.params(i).weightDecay = 0;
       end
   end
end