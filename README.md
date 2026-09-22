# CAN-FLOW: Flow-based conditional cardiac anatomy generation for virtual cohorts
![](can-flow_visual.png)

This repository is the official implementation of the paper
> **"Flow-based conditional cardiac anatomy generation for virtual cohorts"**
> 
> *Konstantinos Kevopoulos, Beatrice Moscoloni, Benjamin Alheit, Cameron Beeche, Julio A. Chirinos, Alexander Heinlein, Mathias Peirlinck*

CAN-FLOW is a **conditional generative model of cardiac anatomy**, based on normalizing flows and diffeomorphisms. This generative modeling approach consists of two steps. 
In the first step, CAN-FLOW learns geometry-only latent representations of diffeomorphic cardiac shape momenta. As a second step, the sex-, age-, and body-mass-index-dependent distribution of those latent representations
is modelled with a conditional normalizing flow. We refer to the sex, age, and BMI conditioning information as **the metadata**. 

When using, please cite:

~~~bibtex
@article{kevopoulos2026flow,
  title={Flow-based conditional cardiac anatomy generation for virtual cohorts},
  author={Kevopoulos, Konstantinos and Moscoloni, Beatrice and Alheit, Benjamin and Beeche, Cameron and Chirinos, Julio A and Heinlein, Alexander and Peirlinck, Mathias},
  doi={10.48550/arXiv.2608.09460}
  journal={arXiv preprint arXiv:2608.09460},
  year={2026}
}
~~~
## Contents

This repository contains:
- The architecture of the CAN-FLOW generative model, and the cVAE used as a baseline for comparison (for details, we refer the reader to the [manuscript](https://arxiv.org/pdf/2608.09460))
- Illustrative script to showcase how to train CAN-FLOW
- Illustrative script to showcase how to generate synthetic cardiac anatomies according to sex, age, and BMI, using a trained instance of CAN-FLOW

The scripts assume user access to a dataset of cardiac anatomies, and the corresponding metadata. 
The training script is designed to be run **after** [Deformetrica's](https://gitlab.com/icm-institute/aramislab/deformetrica/) LDDMM-based anatomical mapping, **which results in the momenta representation of cardiac anatomies**.
This is because the CAN-FLOW autoencoder is designed to operate on this momenta representation. 
CAN-FLOW generates synthetic momenta that should be transformed to synthetic anatomies using **Deformetrica's *geodesic shooting* process**. This step has to be performed by the user and is not included in the repository. 


To illustrate the training and generation workflow of CAN-FLOW, we construct a **synthetic, non-realistic dataset** by sampling from the principal components of biventricular anatomy provided by the [**Cardiac Atlas Project**](https://www.cardiacatlas.org/). More details on the PCA model and the corresponding principal components can be found in [https://www.cardiacatlas.org/biventricular-modes/](https://www.cardiacatlas.org/biventricular-modes/).

Each sampled synthetic anatomy is also assigned an **artificial combination of metadata characteristics, including sex, age, and BMI**. These anatomy–metadata pairs are not intended to represent physiologically realistic subjects. Instead, they are used solely to demonstrate how CAN-FLOW can be trained and subsequently used for conditional generation when a user has access to a real dataset of biventricular cardiac anatomies and associated metadata.

## How to use?

## Contact

For questions, suggestions, or collaborations, please contact:

Konstantinos Kevopoulos                                                                                                                                                                                                              
Department of BioMechanical Engineering, Delft University of Technology, Delft, The Netherlands  
📧 [k.kevopoulos@tudelft.nl](mailto:k.kevopoulos@tudelft.nl)

Mathias Peirlinck  
Department of BioMechanical Engineering, Delft University of Technology, Delft, The Netherlands  
mplab-me at tudelft nl
