function train_network(imdb_video)
	env_paths_training();
    startup();
    opts.gpus = 1;

    if nargin < 1
	    imdb_video = [];
    end
    
    opts.exemplarSize = 127;
   	opts.train.numEpochs = 3;

	my_experiment(imdb_video, opts);
end