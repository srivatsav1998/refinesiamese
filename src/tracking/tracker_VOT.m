function tracker_VOT(varargin)

    cleanup = onCleanup(@() exit() ); % Tell Matlab to exit once the function exits
    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock))); % Set random seed to a different value every time as required by the VOT rules

    %% default hyper-params for SiamFC tracker.
    p.join.method = 'corrfilt';
    p.net = 'cfnet-conv2_e80.mat';
    % p.net_gray = 'cfnet-conv2_gray_e40.mat';
    p.numScale = 3;
    p.scaleStep = 1.0575;
    p.scalePenalty = 0.9780;
    p.scaleLR = 0.52;
    p.responseUp = 8;
    p.wInfluence = 0.2625; % influence of cosine window for displacement penalty
    p.minSFactor = 0.2;
    p.maxSFactor = 5;
    p.zLR = 0.005; % update rate of the exemplar for the rolling avg (use very low values <0.015)
    % p.video = '';
    p.visualization = false;
    p.gpus = 1;
    % p.track_lost = [];
    % p.startFrame = 1;
    p.fout = -1;
    % p.imgFiles = [];
    % p.targetPosition = [];
    % p.targetSize = [];
    % p.track_lost = [];
    % p.ground_truth = [];
    p.numWarmUpEvals = 0;

    %% params from the network architecture params (TODO: should be inferred from the saved network)
    % they have to be consistent with the training
    p.scoreSize = 33;
    p.totalStride = 4;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;
    % prefix and ids
    p.prefix_z = 'branch1_'; % used to identify the layers of the exemplar
    p.prefix_x = 'branch2_'; % used to identify the layers of the instance
    p.id_score = 'score';
    p.trim_z_branch = {'branch1_'};
    p.trim_x_branch = {'branch2_','join_xcorr','final_adj'};
    p.init_gpu = true;
    % Get environment-specific default paths.
    p.paths = struct();
    p.paths = env_paths_tracking(p.paths);
    p = vl_argparse(p, varargin);

    % network surgeries depend on the architecture    
    switch p.join.method
        case 'xcorr'
            p.trim_x_branch = {'branch2_','join_xcorr','final_'};
            p.trim_z_branch = {'branch1_'};
            p.exemplarSize = 127;
            p.instanceSize = 255;
        case 'corrfilt'
            p.trim_x_branch = {'br2_','join_xcorr','fin_adjust'};
            p.trim_z_branch = {'br1_','join_cf','join_crop_z'};
            p.exemplarSize = 255;
            p.instanceSize = 255;
        otherwise
            error('network type unspecified');
    end

    % Load ImageNet Video statistics
    stats = load(p.paths.stats);

    [net_parts{1:2}] = prepare_network_parts(p.net, p);
    % if ~isempty(p.net_gray)
    %     [net_parts_gray{1:2}] = prepare_network_parts(p.net_gray, p);
    % end

    % windowing to penalize large displacements
    window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
    window = window / sum(window(:));
    
    [handle, im_file, region] = vot('polygon');
    [cx, cy, w, h] = get_axis_aligned_BB(region);
    region = [cx, cy, w, h];
    p.targetPosition = region([2, 1]);
    p.targetSize = region([4, 3]);
    
    im = single(imread(im_file));
    % % Switch network if image is gray.
    % if size(im, 3) == 1 && ~isempty(p.net_gray)
    %     net_parts = net_parts_gray;
    % end
    [net_z, net_x] = net_parts{:};
    if ~isempty(p.gpus)
        im = gpuArray(im);
    end
    % if grayscale repeat one channel to match filters size
    if(size(im, 3)==1), im = repmat(im, [1 1 3]); end

    % Init visualization
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end

    avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

    wc_z = p.targetSize(2) + p.contextAmount*sum(p.targetSize);
    hc_z = p.targetSize(1) + p.contextAmount*sum(p.targetSize);
    s_z = sqrt(wc_z*hc_z);
    s_x = p.instanceSize/p.exemplarSize * s_z;
    scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));
    
    scaledExemplar = s_z .* scales;
    % initialize the exemplar
    [z_crop, ~] = make_scale_pyramid(im, p.targetPosition, scaledExemplar, p.exemplarSize, avgChans, stats, p);
    z_crop = z_crop(:,:,:,ceil(p.numScale/2));

    if p.subMean
        z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
    end

    net_z.eval({'exemplar', z_crop});
    z_out_id = net_z.getOutputs();
    z_out_val = get_vars(net_z, z_out_id);
    
    min_s_x = p.minSFactor*s_x;
    max_s_x = p.maxSFactor*s_x;
    min_s_z = p.minSFactor*s_z;
    max_s_z = p.maxSFactor*s_z;

    tot_eval_time = 0;
    tot_z_time = 0;
    
    switch p.join.method
        case 'corrfilt'
            p.id_score = 'join_out';
            % Extract scores for join_out (pre fin_adjust to have them in the range 0-1)
            net_x.vars(end-1).precious = true;
    end
    
    %% Tracker main loop
    scoreId = net_x.getVarIndex(p.id_score);
    overall_tic = tic;
    t = 1;
    while true
        if t > 1
            [handle, im_file] = handle.frame(handle);
            if isempty(im_file)
                break;
            end
            im = single(imread(im_file));
            if ~isempty(p.gpus)
                im = gpuArray(im);
            end
            % if grayscale repeat one channel to match filters size
            if(size(im, 3)==1), im = repmat(im, [1 1 3]); end
            scaledInstance = s_x .* scales;
            % update instance with crop at new frame and previous position
            [x_crops, pad_masks_x] = make_scale_pyramid(im, p.targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
            z_out = interleave(z_out_id, copy(z_out_val, p.numScale));
            [newTargetPosition, newScale] = tracker_step(net_x, s_x, s_z, scoreId, z_out, x_crops, pad_masks_x, p.targetPosition, window, p);
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));

            % update target position
            p.targetPosition = gather(newTargetPosition);

            % update target bbox
            scaledTarget = [p.targetSize(1) .* scales; p.targetSize(2) .* scales];
            p.targetSize = (1-p.scaleLR)*p.targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            % report box to toolkit
            region = [p.targetPosition([2,1]) - 0.5*p.targetSize([2,1]), p.targetSize([2,1])];
            handle = handle.report(handle, region);

            % update the exemplar with crop at new frame and new position
            if p.zLR > 0
                scaledExemplar = s_z .* scales;
                [z_crop, ~] = make_scale_pyramid(im, p.targetPosition, scaledExemplar, p.exemplarSize, avgChans, stats, p);
                z_crop = z_crop(:,:,:,ceil(p.numScale/2));
                if p.subMean,   z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3])); end
                eval_z_tic = tic;
                net_z.eval({'exemplar', z_crop});
                eval_z_time = toc(eval_z_tic);
                tot_z_time = tot_z_time+eval_z_time;
                z_out_val_new = get_vars(net_z, z_out_id);
                
                % template update with rolling average                                                
                update = @(curr, next) (1-p.zLR) * curr + p.zLR * next;
                z_out_val = arrayfun(@(i) update(z_out_val{i}, z_out_val_new{i}), ...
                             1:numel(z_out_id), 'UniformOutput', false);
                                                                  
                s_z = max(min_s_z, min(max_s_z, (1-p.scaleLR)*s_z + p.scaleLR*scaledExemplar(newScale)));
            end
        else
            % at the first frame output position and size passed as input (ground truth)
        end
        t = t + 1;

        rectPosition = [p.targetPosition([2,1]) - p.targetSize([2,1])/2, p.targetSize([2,1])];
        if p.visualization
            if isempty(videoPlayer)
                figure(1), imshow(im/255);
                figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
                drawnow
                % fprintf('Frame %d\n', p.startFrame+i);
            else
                im = gather(im)/255;
                im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
            end
        end
    end

    % overall_time = toc(overall_tic);
    % n_frames_ontrack = sum(sum(bboxes==0,2)~=4);
    % if isempty(p.track_lost)
    %     speed = (nImgs-p.startFrame+1) / overall_time;
    % else
    %     speed = n_frames_ontrack/overall_time;
    % end
    handle.quit(handle);
end


function [net_z, net_x] = prepare_network_parts(net, p)
    %% Load pre-trained network
    if ischar(net)
        % network has been passed as string
        net_path = fullfile(p.paths.net_base, net);
        net_z = load(net_path,'net');
    else
        % network has been passed as object
        net_z = net;
    end
        
    net_x = net_z;
    % Load a second copy of the network for the second branch
    net_z = dagnn.DagNN.loadobj(net_z.net);
    
    % Sanity check
    switch p.join.method
        case 'xcorr'
            assert(~find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
        case 'corrfilt'
            assert(find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
    end
    % create a full copy of the network, not just of the handle
    net_x = dagnn.DagNN.loadobj(net_x.net);

    net_z = init_net(net_z, p.gpus, p.init_gpu);
    net_x = init_net(net_x, p.gpus, false);
    % visualize net before trimming
    % display_net(net_z, {'exemplar', [255 255 3 8], 'instance', [255 255 3 8]}, 'net_full')

    %% Divide the net in 2
    % exemplar branch (only once per video) computes features for the target
    for i=1:numel(p.trim_x_branch)
        remove_layers_from_prefix(net_z, p.trim_x_branch{i});
    end
    % display_net(net_z, {'exemplar', [255 255 3 8]}, 'z_net')
    % display_net(net_z, {'exemplar', [127 127 3 8], 'target', [6 6 1 8]}, 'z_net')
    % instance branch computes features for search region and cross-correlates with z features
    for i=1:numel(p.trim_z_branch)
        remove_layers_from_prefix(net_x, p.trim_z_branch{i});
    end
    % display_net(net_x, {'instance', [255 255 3 8], 'br1_out', [30 30 32 8]}, 'x_net')
    % display_net(net_x, {'instance', [255 255 3 8], 'join_tmpl_cropped', [17 17 32 8]}, 'x_net')

    % Warm up the network.
    if ~isempty(p.gpus)
        z_out_id = net_z.getOutputs();
        for i = 1:p.numWarmUpEvals
            z_crop = gpuArray(single(randn(p.exemplarSize, p.exemplarSize, 3)));
            net_z.eval({'exemplar', z_crop});
            x_crops = gpuArray(single(randn(p.instanceSize, p.instanceSize, 3, p.numScale)));
            z_out_val = get_vars(net_z, z_out_id);
            z_out = interleave(z_out_id, copy(z_out_val, p.numScale));
            net_x.eval([z_out, {'instance', x_crops}]);
        end
    end
end


function vals = get_vars(net, ids)
    vals = cellfun(@(id) net.getVar(id).value, ids, 'UniformOutput', false);
end

function u = copy(v, n)
    u = cellfun(@(x) repmat(x, [1 1 1 n]), v, 'UniformOutput', false);
end
