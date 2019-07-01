function rename_baseline(net)
%RENAME_BASELINE Summary of this function goes here
%   Detailed explanation goes here
    nlayers = numel(net.layers);
    for i=1:nlayers
        lname = net.layers(i).name;
        newlname = '';
        parts = strsplit(lname, '_');
        if strcmp(parts{1},'br1')
            parts{1} = 'branch1';
            newname = parts{1};
            for j=2:numel(parts)
                newname = [newname '_' parts{j}];
            end
            newlname = newname;
        end
        if strcmp(parts{1}, 'br2')
            parts{1} = 'branch2';
            newname = parts{1};
            for j=2:numel(parts)
                newname = [newname '_' parts{j}];
            end
            newlname = newname;
        end
        if ~isempty(newlname)
            net.renameLayer(lname, newlname);
        end
    end
end

