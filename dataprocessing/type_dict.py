

type_dict = {
    # dyno
    'dentate':'UMI', 
    'immune':'UMI', 
    'neonatal':'UMI', 
    'planaria_full':'UMI', 
    'planaria_muscle':'UMI',
    'aging':'non-UMI', 
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI', 
    'germline':'non-UMI',    
    'human':'non-UMI', 
    'mesoderm':'non-UMI',
    
    # dyngen
    'bifurcating_2':'non-UMI',
    "cycle_1":'non-UMI', 
    "cycle_2":'non-UMI', 
    "cycle_3":'non-UMI',
    "linear_1":'non-UMI', 
    "linear_2":'non-UMI', 
    "linear_3":'non-UMI', 
    "trifurcating_1":'non-UMI', 
    "trifurcating_2":'non-UMI', 
    "bifurcating_1":'non-UMI', 
    "bifurcating_3":'non-UMI', 
    "converging_1":'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}
source_dict = {
    'dentate':'dyno', 
    'immune':'dyno', 
    'neonatal':'dyno', 
    'planaria_muscle':'dyno',
    'planaria_full':'dyno',
    'aging':'dyno', 
    'cell_cycle':'dyno',
    'fibroblast':'dyno', 
    'germline':'dyno',    
    'human':'dyno', 
    'mesoderm':'dyno',
    
    'bifurcating_2':'dyngen',
    "cycle_1":'dyngen', 
    "cycle_2":'dyngen', 
    "cycle_3":'dyngen',
    "linear_1":'dyngen', 
    "linear_2":'dyngen', 
    "linear_3":'dyngen', 
    "trifurcating_1":'dyngen', 
    "trifurcating_2":'dyngen', 
    "bifurcating_1":'dyngen', 
    "bifurcating_3":'dyngen', 
    "converging_1":'dyngen',
    
    'linear':'our model',
    'bifurcation':'our model',
    'multifurcating':'our model',
    'tree':'our model',
}


__all__=[type_dict]