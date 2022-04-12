## raisimGymTorch_treadmill

<img src=./images/tensorboard_log_update.png width=400>
<img src=./images/humanoid2d_image.png width=200>

## Modified the raisimGymTorch in raisimlib 1.0

### put it in parallel with the raisimGymTorch as below
raisimlib/raisimGymTorch  
raisimlib/raisimGymTorchBio

### Run

1. Compile raisimgym in raisimlib/raisimGymTorchBio: 
```python ./setup.py develop --user --prefix= --CMAKE_PREFIX_PATH "../../raisim_build"```

2. run runnerMod.py of the task for training (for humanoid2d example): 
```python ./raisimGymTorch/env/envs/rsg_humanoid2d/runnerMod.py --mode "train"```

3. run runnerMod.py of the task for testing (for humanoid2d example): 
```python ./raisimGymTorch/env/envs/rsg_humanoid2d/runnerMod.py --mode "test" --test_modelfile_name "FULL_PATH_NAME_YOUR_PPO_MODEL.pt" --test_envscaledir_name "FULL_PATH_YOUR_MEAN_AND_VAR_LOG_DIR" --test_envscale_niter "CHOOSE_ITERATION_ID" --total_timesteps 4000```

### Debugging
1. Compile raisimgym with debug symbols: ```python setup develop --Debug```. This compiles <YOUR_APP_NAME>_debug_app
2. Run it with Valgrind. I strongly recommend using Clion for 

### Todo
1. Saving and loading the torch model for retraining and testing  
