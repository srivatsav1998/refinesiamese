%% Sample execution for Baseline-conv5 (improved Siam-FC)
% hyper-parameters reported in Supp.material for CVPR'17, Table 2 for arXiv version
tracker_par.join.method = 'xcorr';
tracker_par.net = 'net-epoch-3.mat';
% tracker_par.net_gray = 'baseline-conv5_gray_e100.mat';
tracker_par.scaleStep = 1.0470;
tracker_par.scalePenalty = 0.9825;
tracker_par.scaleLR = 0.68;
tracker_par.wInfluence = 0.175;
tracker_par.zLR = 0.0102;

% tracker_par.scaleStep = 1.0470;
% tracker_par.scalePenalty = 1.0425;
% tracker_par.scaleLR = 0.60;
% tracker_par.wInfluence = 0.175;
% tracker_par.zLR = 0.125;

[~,~,dist_rfs,overlap_rfs,~,~,~,~] = run_tracker_evaluation('all', tracker_par);
% [~,~,dist,overlap,~,~,~,~] = run_tracker_evaluation('vot2013_iceskater', tracker_par);
% [~,~,dist,overlap,~,~,~,~] = run_tracker_evaluation('tc_Basketball_ce3', tracker_par);
% [~,~,dist,overlap,~,~,~,~] = run_tracker_evaluation('tc_Ball_ce4', tracker_par);