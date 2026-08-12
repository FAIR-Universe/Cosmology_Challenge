# Data
***

Participants will work with simulated datasets mimicking observations from the [<ins>Hyper Suprime-Cam (HSC) survey</ins>](https://science.jpl.nasa.gov/projects/hyper-suprime-cam/). Each data is a 2D image of dimension $1424 \times 176$, corresponds to the convergence map of redshift BIN $2$ of WIDE12H subfield in HSC Y3, pixelized with a resolution of $2$ arcmin. 

These weak lensing convergence maps are generated from high-resolution cosmological ray-tracing simulations with $101$ different spatially-flat $\Lambda \text{CDM}$ cosmological models. Each cosmological model differs in cosmological parameters $\Omega_m$, the fraction of the total matter density of the Universe, and $S_8$, the amplitude of matter fluctuations on $8 \,\mathrm{Mpc}/h$ scales in the Universe today. These two parameters serve as the label of each data. 

In addition to the cosmological signal, we also model various realistic systematic effects (distortions to the data), such as baryonic effect and photometric redshift uncertainty. These systematics are introduced in the data generation process, which we fully sampled in the training set so that the participants can marginalize over them. The parameters corresponding to these systematic models are nuisance parameters and need to be marginalized during inference.

We have prepared the training data and the public test data for participants to develop their methods that will be scored on Codabench. Please download them from
[**<ins>Training Data / Test Data (8.7 GB)</ins>**](https://www.codabench.org/datasets/download/09b7a27a-bfb3-4031-a5b0-9b3c533f45e6/).

In the downloaded file, you will find:
- **`WIDE12H_bin2_2arcmin_kappa_newrealization.npy`**: The training data of this competition. It contains $101 \times 256$ realizations of noiseless convergence maps. In this competition, we refer to **in-distribution (InD)** data as any data generated using the same simulation setup and the same distributions of cosmological and physical parameters as the training data.

    ⚠️ **NOTE:** The training data used here are generated from new simulations with no shared random seeds across all $101 \times 256$ realizations, in contrast to the Phase-1 training data, which were produced using only $256$ random seeds. Participants are allowed to include the Phase-1 (InD) training data into their training data, but they should be aware that the limited set of shared random seeds in Phase 1 may introduce correlations that are absent in the new simulations, and thus care must be taken to avoid potential biases or overfitting arising from this difference.

- **`WIDE12H_bin2_2arcmin_mask.npy`**: The mask that marks which regions of the convergence maps are included and which are excluded in the real cosmological survey.

- **`label_newrealization.npy`**: Although not essential for this competition, we also provide the cosmological and physical parameters used for training data generation. The cosmological and physical parameters include:
    - $\Omega_m$: a parameter that represents the fraction of the matter energy density in the universe. 
    - $S_8$: a parameter that quantifies the amplitude of matter fluctuations in the present-day universe.
    - $T_{\rm{AGN}}$: A baryonic physics parameter that controls how strongly AGN feedback suppresses small-scale structure.
    - $f_0$: A baryonic physics parameter that controls the overall amplitude of baryonic effects associated with star formation.    
    - $\Delta_z$: A photometric redshift uncertainty parameter that quantifies the systematic bias in photometric redshift estimation.

    The figure below shows some examples of the training data and how they are varied with different physical parameters and pixel-level noise.
<center>
<img src="image-1.png" width="400"> 
</center><br>

- **`WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy`**: 
The public test dataset for this competition contains 10,000 noisy convergence maps. A subset of these samples is generated under alternative physical models and is therefore considered out-of-distribution (OoD), while the remaining samples are in-distribution (InD). The OoD samples are labeled as $y_i=1$, and the InD samples as $y_i=0$. However, participants are not provided with the ground-truth labels, any explicit OoD examples, or details regarding how the OoD data are generated. Participants are required to submit their predictions on the public test set to Codabench (see `Evaluation` and `Starting Kit` tabs for more details).  

    ⚠️ **Note 1:** On August 12 2026, the test dataset of this competition has been updated to this `*_test_phase2_new_v2.npy` version to fix some numerical problems. The `*_test_phase2_new.npy` version that was previously released by this competition is no longer valid, and you should only use the `*_test_phase2_new_v2.npy` file provided here for further submissions.

    ⚠️ **Note 2:** The test dataset can only be used as inputs for inference. It should **NOT** be used to collectively to fit, train, adapt, cluster, calibrate, normalize, rank, or modify the model. For more details, please refer to the **Rules for OoD Detection Methods** section in `Overview`.