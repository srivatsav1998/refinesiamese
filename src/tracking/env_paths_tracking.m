function paths = env_paths_tracking(varargin)
    paths.net_base = 'G:\srivatsav\experiment\refineSiamese\src\training\trained_data\without baseline learning\arch1\5 per video\'; % e.g. '/home/luca/cfnet/networks/';
    paths.eval_set_base = 'G:\srivatsav\experiment\cfnet\data\'; % e.g. '/home/luca/cfnet/data/';
    paths.stats = 'G:\srivatsav\experiment\refineSiamese\data\imdb_stats.mat'; % e.g.'/home/luca/cfnet/data/ILSVRC2015.stats.mat';
    paths = vl_argparse(paths, varargin);
end
