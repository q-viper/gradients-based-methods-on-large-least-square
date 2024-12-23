# gradients-based-methods-on-large-least-square
Some codes and experiments I did as a part of Master's seminar titled as `Comparing Different Gradient Based Schemes on Large Least Squared Problems`.

## What did I do?
I explored different gradient descent based parameter update techniques i.e. Optimizers on Least Squared Problems. There were lots of factor to compare optimizers and I mostly did based on the learning rate. i.e. constant LR based optimizers and adaptive LR based optimizers. I made a little suplotting and visualizer on top of Matplotlib and it is nice.

### Workflow
1. Define a DataConfig from [gradls/datagenerator.py](gradls/datagenerator.py)
2. Define data from `DataGenerator`.
3. For a single experiment:

```python
viz_config = MatplotlibVizConfig(figsize=(15,10),title="My Exp",
                                 use_tex=False, sharey=False)
exp_config = ExperimentConfig(name="Exp5", loss=LossType.MSE, viz_config=viz_config, 
                              num_epochs=1000, batch_size=1,learning_rate=0.1,momentum=0.9, optimizer=Optimizer.NESTEROV,
                              model=None, metrics=[LossType.MSE],
                              log_every=1, log_anim=False, anim_fps=10, plot_format='png')

data = DataGenerator(DataGeneratorConfig(num_rows=1000, num_cols=5, noise_norm_by=10))

exp = Experiment(config=exp_config)
exp.load_data(data)
```

4. But I had to run 100s of experiments so had to use multiprocessing. Hence there is `ExperimentHandler` as well.

```python
expt_handler_config = ExperimentHandlerConfig(root_dir=Path("expt_test"), max_experiments=-1, num_jobs=10,
                                              data_gen=[DataGenerator(DataGeneratorConfig(num_rows=1000, num_cols=5, noise_norm_by=10))], 
                                              losses=[LossType.MAE, LossType.MSE], metrics=[LossType.MAE, LossType.MSE],
                     num_epochs=[100], batch_sizes=[1, 16, 32, -1], optimizers=[Optimizer.SGD],
                     seeds=[100], learning_rates=[0.1, 0.01,0.001,0.0001], momentums=[0.1, 0.9], plot_format='pdf', log_anim=False, log_plots=True, log_real_params=False)
expt_handler = ExperimentHandler(expt_handler_config.root_dir, expt_handler_config)
```

Above configuration makes experiment like in step 3 for each of the possible cases in `ExperimentHandlerConfig`. It will be like `number of losses * number of data gen * number of batch sizes * number of optimizers * number of seeds * number of learning rates * number of momentums`. In above case: nearly `64 experiments`. They will be run in 10 processes block and once all of the 10 jobs finishes, another batch takes the place. This is handled in `ExperimenHandler.make_experiments` by saving the experiment and related configs and loading back in `experiment_proc.py`. And following config generated and ran all the experiments:

```python

expt_handler_config = ExperimentHandlerConfig(root_dir=Path("expt_res_less_noise"), max_experiments=-1, num_jobs=10,
                                              data_gen=[DataGenerator(DataGeneratorConfig(num_rows=1000, num_cols=10, noise_norm_by=10))], losses=[LossType.MAE, LossType.MSE], metrics=[LossType.MAE, LossType.MSE],
                     num_epochs=[1000], batch_sizes=[1, 16, 32, -1], optimizers=[Optimizer.SGD, Optimizer.MOMENTUM, Optimizer.NESTEROV, 
                                                                               Optimizer.ADAM, Optimizer.RMSPROP, Optimizer.ADAGRAD, Optimizer.ADADELTA],
                     seeds=[100], learning_rates=[0.1, 0.01,0.001,0.0001], momentums=[0.1, 0.9], plot_format='pdf', log_anim=False, log_plots=True, log_real_params=False)
expt_handler = ExperimentHandler(expt_handler_config.root_dir, expt_handler_config)

expt_handler.make_experiments()
```

## To generate plots:
Please follow [notebooks/experiment_plots.ipynb](notebooks/experiment_plots.ipynb). It starts from the **All Plots** section. Steps:
1. Load experiments using `ExperimentHandler`.
2. Plot using `MatplotlibVisualizer`


## Reproducing
All of the results could be reproduced from the notebooks.
* [notebooks/experiment.ipynb](notebooks/experiment.ipynb)
* [notebooks/experiment_plots.ipynb](notebooks/experiment_plots.ipynb)

## Results
* [Final Report](assets/gradient_based_methods_in_ls_final_handout.pdf)
* Plots are also in [assets](assets/)


