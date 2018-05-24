CONSNet: Convolutional Neural Networks for Skull-stripping in Brain MR Imaging using Consensus-based Silver Standard Masks
==========================================================================================================================

MIT License
Copyright (c) 2018 Medical Image Computing Lab - MICLab Unicamp


## Description

This is an alpha version of the CONSNet presented in the paper *Convolutional Neural Networks for Skull-stripping in Brain MR Imaging using Consensus-based Silver Standard Masks* . The article is available for download here: https://arxiv.org/pdf/1804.04988.pdf . If you use this code on your work, please cite the this article.

- Lucena, Oeslle, et al. "Convolutional Neural Networks for Skull-stripping in Brain MR Imaging using Consensus-based Silver standard Masks." arXiv preprint arXiv:1804.04988 (2018).

@article{lucena2018convolutional,
  title={Convolutional Neural Networks for Skull-stripping in 
  Brain MR Imaging using Consensus-based Silver standard Masks},
  author={Lucena, Oeslle and Souza, Roberto and Rittner, Leticia and 
  Frayne, Richard and Lotufo, Roberto},
  journal={arXiv preprint arXiv:1804.04988},
  year={2018}
}


## Requirements
- Python 2.7 (working in progress to be compatible to Python 3.0)
- NumPy 1.14.3
- SciPy 1.0.1 
- Sklearn 0.19.1
- Nibabel 2.2.1 
- Keras 2.1.6
- Tensorflow 1.4.0

## Docker image (TO DO)


## Usage

### Inference using CONSNet pre-trained models
If you just to want to run the inference for a new volume, you need to run the command below providing a text file with the path of the respective data. 

```
python infer.py -input input_data.txt
```

### Training your own model
You can train your own model for CONSNet. You need to provide two text files with the data and the respective annotated masks paths. Also, feel free to change the default parameters.

```
python prep-train.py -tr_original_filename data.txt -tr_consensus_filename mask.txt

```

### Fine-tuning CONSNet (TO DO)



## Contact

If you have any doubts, questions or suggestions to improve this code, please contact me at: oeslle.lucena.souza@gmail.com
