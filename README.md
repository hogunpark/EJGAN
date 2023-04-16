# Incorporating Experts' Judgment into Machine Learning Models



This repository provides a reference implementation of *EJ-GAN* as described in the paper:<br>
> Incorporating Experts' Judgment into Machine Learning Models<br>
> Hogun Park, Aly Megahed, Peifeng Yin, Yuya Ong, Pravar Mahajan, Pei Guo <br>
> Expert Systems with Applications, Elsevier, 2023.<br>
> 
We note that our code was tested under Python 3.6.10.

## Usage for running GAN-based Conflict Resolution Function
#### Example
To run *EJ-GAN* on CREDIT dataset, execute the following command from the project home directory:<br/>


    usage: main_GAN_Classification.py [-h] --input INPUT
                                  [--export_plot EXPORT_PLOT] [--prior PRIOR]
                                  [--weight WEIGHT] [--num-steps NUM_STEPS]
                                  [--hidden-size HIDDEN_SIZE]
                                  [--dc_weight DC_WEIGHT]
                                  [--batch-size BATCH_SIZE]
                                  [--log-every LOG_EVERY] [--n_bins N_BINS]
                                  [--classifier CLASSIFIER]


 			e.g., --input "data/credit/input.csv" --prior 4 --classifier 0  # 4 was used for prior in our experiment

## Citing
If you find *EJ-GAN* useful for your research, please consider citing the following paper:

	@article{ejgan-eswa,
	 author = {Park, Hogun and Megahed, Aly and Yin, Peifeng and Ong, Yuya and Mahajan, Pravar and Guo, Pei},
	 title = {Incorporating Experts' Judgment into Machine Learning Models},
	 journal ={Expert Systems with Applications},
	 year = {2023},
     publisher={Elsevier}
	}


### ACKNOWLEDGEMENT

The implementation of this repository is porked and based on the [codebase](https://github.com/IBM/hybrid-expert-intuition-model). </br>
@Copyright 2020 IBM Corporation</br>
Licensed under the Apache License, Version 2.0 (the "License").</br>
You may not use this file except in compliance with the License.</br>
