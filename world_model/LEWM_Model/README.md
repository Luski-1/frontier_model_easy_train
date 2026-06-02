```text
\LEWM_MODEL
│  DATA_EXAMPLE.md
│  lewm.md
│  README.md
│  SIGReg代码.md
│  SIGReg理论.md
│  tree.txt
│  
├─mini_lewm
│  │  dataset.py
│  │  eval.py
│  │  loss.py
│  │  model.py
│  │  train.py
│  │  utils.py
│  │  
│  └─configs
│          eval.yaml
│          train.yaml
│          
├─reference
│  │  eval.py
│  │  jepa.py
│  │  LICENSE
│  │  module.py
│  │  README.md
│  │  stable_worldmodel.data.dataset.py
│  │  stable_worldmodel.data.formats.hdf5.py
│  │  train.py
│  │  utils.py
│  │  
│  ├─assets
│  │      lewm.gif
│  │      
│  └─config
│      ├─eval
│      │  │  cube.yaml
│      │  │  pusht.yaml
│      │  │  reacher.yaml
│      │  │  tworoom.yaml
│      │  │  
│      │  ├─launcher
│      │  │      local.yaml
│      │  │      
│      │  └─solver
│      │          adam.yaml
│      │          cem.yaml
│      │          
│      └─train
│          │  lewm.yaml
│          │  
│          ├─data
│          │      dmc.yaml
│          │      ogb.yaml
│          │      pusht.yaml
│          │      tworoom.yaml
│          │      
│          ├─launcher
│          │      local.yaml
│          │      
│          └─model
│                  lewm.yaml
│                  
└─train_result
        env_49.mp4
        pusht_results.txt
```