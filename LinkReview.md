# LinkReview

- Here we collect all the works that may be useful for writing our paper

> [!NOTE]
> This review table will be updated, so it is not a final version.

## Papers and code

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| Main Articles | Siamese network features for image matching | 2016 | Melekhov I., Kannala J., Rahtu E. | [IEEE](https://ieeexplore.ieee.org/document/7899663) |  | Default approach, loss function ideas |
|  | A simple framework for contrastive learning of visual representations | 2020 | Chen T. et al. | [arXiv](https://arxiv.org/abs/2002.05709) |  | Default approach, better loss function |
|  | Learning Transferable Visual Models from Natural Language Supervision | 2021 | Radford, A. et al. | [arXiv](https://arxiv.org/abs/2103.00020) |  | CLIP, Image + text, may not be useful in our case |
|  | Barlow Twins: Self-Supervised Learning via Redundancy Reduction | 2021 | Zbontar, J. et al. | [arXiv](https://arxiv.org/abs/2103.03230) |  | Barlow Twins, main model for the first approach |
| Useful | VISSL |  | Facebook |  | [GitHub](https://github.com/facebookresearch/vissl) | Computer **VI**sion library for state-of-the-art **S**elf-**S**upervised **L**earning research with PyTorch |
|  | MMSelfSup |  | OpenMMLab |  | [GitHub](https://github.com/open-mmlab/mmselfsup) | Collection of pretrained models (including SimCLR, Barlow Twins) |
|  | Contrastive Learning Walkthrough |  | v7labs | [WebPage](https://www.v7labs.com/blog/contrastive-learning-guide) |  | Big collection of theoretical and practical results |


## Datasets

| Content | Link | Description |
| :--- | :---: | :--- |
| Cell Datasets | [CIL](https://www.cellimagelibrary.org/home) | Cell datasets under some copyright terms |
| Haxby fMRI Dataset | [Haxby](http://data.pymvpa.org/datasets/haxby2001) | fMRI scans after showing patients greyscale images |
| Body Parts MRI | [VisibleHuman](https://datadiscovery.nlm.nih.gov/Images/Visible-Human-Project/ux2j-9i9a/about_data?_gl=1*13eefcd*_ga*MTIxODMwNzQ5NC4xNzQxMjU2NDU3*_ga_7147EPK006*MTc0MTI1NjQ1Ny4xLjEuMTc0MTI1NjU1My4wLjAuMA..*_ga_P1FPTH9PL4*MTc0MTI1NjQ1Ny4xLjEuMTc0MTI1NjU1My4wLjAuMA..) | MRI of different human parts including head, arms, legs, etc. |
