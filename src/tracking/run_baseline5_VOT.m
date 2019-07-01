function run_baseline5_VOT()

startup;

%% Sample execution for Baseline-conv5 (improved Siam-FC)
% hyper-parameters reported in Supp.material for CVPR'17, Table 2 for arXiv version
tracker_par.join.method = 'xcorr';
tracker_par.net = 'my_model.mat';
% tracker_par.net_gray = 'baseline-conv5_gray_e100.mat';
tracker_par.scaleStep = 1.0470;
tracker_par.scalePenalty = 0.9825;
tracker_par.scaleLR = 0.68;
tracker_par.wInfluence = 0.175;
tracker_par.zLR = 0.0102;

% Uncomment the following line to enable CPU mode.
% tracker_par.gpus = [];

tracker_par.numWarmUpEvals = 1;
tracker_VOT(tracker_par);

end
