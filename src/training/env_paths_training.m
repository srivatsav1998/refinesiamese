function opts = env_paths_training(opts)

    opts.rootDataDir = 'G:\srivatsav\curated\ILSVRC2015\Data\VID\train\'; % where the training set is
    opts.imdbVideoPath = 'G:\srivatsav\experiment\refineSiamese\data\imdb_video.mat'; % where the training set metadata are
    opts.imageStatsPath = 'G:\srivatsav\experiment\refineSiamese\data\imdb_stats.mat'; % where the training set stats are

end
