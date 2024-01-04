## LiGAN configuration settings

The training and generation scripts are controlled by configuration files. The configuration settings are documented below.

### generate.config

```
# general settings
out_prefix: <str: common prefix for output files>
model_type: <str: type of generative model architecture>
random_seed: <int: seed for random number generation>
verbose: <bool: use verbose output>
device: <{cuda, cpu}: device to run model>

data: # settings for data loading and gridding
  data_root: <str: path to directory containing receptor/ligand files>
  data_file: <str: path to data file specifying receptor/ligand paths>
  batch_size: <int: number of examples per batch>
  rec_typer: <str: receptor atom typing scheme>
  lig_typer: <str: ligand atom typing scheme>
  use_rec_elems: <bool: use different element set for receptors>
  resolution: <float: grid resolution in angstroms>
  grid_size: <int: number of grid points per dimension>
  shuffle: <bool: randomly shuffle data examples>
  random_rotation: <bool: apply uniform random grid rotations>
  random_translation: <float: maximum random grid translation in angstroms>
  diff_cond_transform: <bool: apply different random transform to conditional branch>
  diff_cond_structs: <bool: use different (rec,lig) structures for conditional branch>

gen_model: # generative model architecture; must be the same as was used for training!
  n_filters: <int: number of filters in first convolution layer>
  width_factor: <int: factor by which to increase # filters in each level>
  n_levels: <int: number of convolution blocks with pooling layers between>
  conv_per_level: <int: number of convolution layers per block/level>
  kernel_size: <int: convolution kernel size>
  relu_leak: <float: slope for leaky ReLU activation function>
  batch_norm: <int: flag for including batch normalization layers>
  spectral_norm: <int: flag for including spectral normalization layers>
  pool_type: <str: type of pooling layers (a=average, m=max, c=conv)>
  unpool_type: <str: type of upsampling layers (n=nearest, c=conv)>
  pool_factor: <int: kernel size and stride for pooling layers>
  n_latent: <int: width of latent space>
  init_conv_pool: <int: flag for including a separate initial conv/pool layer pair>
  skip_connect: <bool: flag for conditional skip connections from encoder to decoder>
  block_type: <str: type of convolution block (c=standard, r=residal, d=dense)>
  growth_rate: <int: growth rate for dense convolution blocks (no effect otherwise)>
  bottleneck_factor: <int: bottleneck factor for dense blocks (no effect otherwise)>
  state: <str: path to generative model state file containing trained weights>

atom_fitting: # atom fitting settings
  beam_size: <int: number of top-ranked structures to maintain during search>
  multi_atom: <bool: allow placing multiple detected atoms simultaneously>
  n_atoms_detect: <int: number of top-ranked atoms to detect at each step>
  apply_conv: <bool: apply convolution with density kernel before detecting atoms>
  threshold: <float: threshold for detecting atoms in residual density>
  peak_value: <float: upper limit for detecting atoms in residual density>
  min_dist: <float: minimum distance between detected atoms>
  apply_prop_conv: <bool: apply convolution to atom property channels>
  interm_gd_iters: <int: number of gradient descent steps after each atom placement>
  final_gd_iters: <int: number of gradient descent steps on final structure>

generate: # molecule generation settings
  n_examples: <int: number of examples from input data to generate>
  n_samples: <int: number of samples to generate per input example>
  prior: <bool: sample from prior distribution instead of posterior>
  var_factor: <float: variability factor; controls sample diversity>
  post_factor: <float: posterior factor; controls similarity to input>
  stage2: <bool: use stage2 VAE; experimental feature>
  truncate: <bool: truncate tails of latent sampling distribution>
  interpolate: <bool: latent interpolation; experimental feature>
  spherical: <bool: use spherical latent interpolation>
  fit_atoms: <bool: fit atoms to generated densities>
  add_bonds: <bool: add bonds to generate molecules from fit atoms>
  uff_minimize: <bool: minimize internal energy of generated molecules>
  gnina_minimize: <bool: minimize generated molecules in receptor pocket>
  minimize_real: <bool: minimize input molecules for comparison>

output:
  batch_metrics: <bool: compute batch-level metrics>
  output_grids: <bool: write generated densities to .dx files (these are large)>
  output_structs: <bool: write fit atom structures to .sdf files>
  output_mols: <bool: write generated molecules to .sdf files>
  output_latents: <bool: write sampled latent vectors to .latent files>
  output_visited: <bool: include all visited atom fitting structures>
  output_conv: <bool: write atom density kernel to .dx files>
```

### train.config

```
out_prefix: <str: set output file prefix of generative model or optimizer>
model_type: <str: switch Architechture of the NN model, in this case we only implemented to CVAE.>
random_seed:  <int: seed for random number generation>
caffe_init: False # Not to use
continue: False # Not to use
max_n_iters: <int: max iteration of training epoches>
balance: False # Not to use
device: <str: specify gpu device (on GOJO, cuda:0 to cuda:2 // on suzuki_local, cuda:0 only)>

data:
  data_root: /mnt/d/Documents_2023/crossdock2020_full/ # Not to use
  batch_size: <int:  training examples utilized in one iteration. commonly use the power of 2>
  rec_typer: oadc-1.0 # Not to use
  lig_typer: oadc-1.0 # Not to use
  use_rec_elems: True # Not to use
  resolution: 0.5 # Not to use
  grid_size: 48 # Not to use
  shuffle: True # Not to use
  random_rotation: True # Not to use
  random_translation: 2.0 # Not to use
  train_file: <path: specify the location of raw full-data>
  test_file: _ # Not to use
  train_ratio: <float: train-test ratio>
  prepared_dataset: <bool: if prepared dataset with pickle, this param should be True>
  train_dataset: <path: specity the location of prepared train dataset>
  test_dataset: <path: location of prepared test dataset>
  cut_size: # Not to use

gen_model:
  n_filters: # Not to use
  width_factor: # Not to use
  n_levels: <int: how many blocks in the architechture>
  conv_per_level: <int: how many layers in a single block>
  kernel_size: 3 # Not to use
  relu_leak: 0.1 # Not to use
  batch_norm: 0 # Not to use
  spectral_norm: <int: determine spectral normalization is required or not>
  pool_type: <a, m: pooling type average or max>
  unpool_type: 'n' # Not to use
  pool_factor: 2 # Not to use
  n_latent: <int: the size of latent space>
  init_conv_pool: 0 # Not to use
  skip_connect: <bool: the model have `skip_connection` or not>
  block_type: r # Not to use
  growth_rate: 0 # Not to use
  bottleneck_factor: 0 # Not to use

disc_model: # Not to use
  n_filters: 0
  width_factor: 2
  n_levels: 0
  conv_per_level: 0
  kernel_size: 3
  relu_leak: 0.1
  batch_norm: 0
  spectral_norm: 1
  pool_type: a
  pool_factor: 2
  n_output: 1
  init_conv_pool: 0
  block_type: r
  growth_rate: 0
  bottleneck_factor: 0

loss_fn:

  types: # FIX THIS CODE
    recon_loss: '2'
    gan_loss: 'w'
    recon2_loss: 'c'

  weights:
    kldiv_loss: <float: the loss weight of ENCODER>
    recon_loss: <float: the loss weight of RECONSTRUCTION>
    steric_loss: 0.0 # Not to use
    gan_loss: 0.0 # Not to use
    kldiv2_loss: 0.0 # Not to use
    recon2_loss: <float: the loss weight of adj matrix decoder>

  schedules:# Not to use

    kldiv_loss:
      start_iter: 450000
      end_wt: 1.6
      period: 200000
      type: d

    recon_loss:
      start_iter: 450000
      end_wt: 4.0
      period: 200000
      type: n
    
    recon2_loss:
      start_iter: 450000
      end_wt: 4.0
      period: 200000
      type: n

  learn_recon_var: 0

gen_optim:
  type: RMSprop # FIX
  lr: <float: learning rate>
  clip_gradient: 5000 # FIX
  n_train_iters: 1 # FIX

disc_optim: # Not to use
  type: RMSprop
  lr: 0.0e+00
  clip_gradient: 0
  n_train_iters: 0

atom_fitting:# Not to use
  beam_size: 1
  multi_atom: False
  n_atoms_detect: 1
  apply_conv: False
  threshold: 0.1
  peak_value: 1.5
  min_dist: 0.0
  apply_prop_conv: False
  interm_gd_iters: 10
  final_gd_iters: 100

train:
  max_iter: <int: epochs to train>
  n_test_batches: 10 # Not to use
  test_interval: <int: the interval of test>
  fit_interval: 100 # Not to use
  norm_interval: 10 # Not to use
  save_interval: <int: the interval of model saving>

wandb:
  use_wandb: <bool: use weight and biases or not>
  out_prefix: <path: location of the wandb output log files>
  mem: out_prefix will be deleted in on-stage run # FIX

```
