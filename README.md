<p align="center">
<h1 align="center"><strong>GScream: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal</strong></h1>
<h3 align="center">ECCV 2024</h3>

<p align="center">
    <a href="https://w-ted.github.io/">Yuxin Wang</a><sup>1</sup>,</span>
    <a href="https://wuqianyi.top/">Qianyi Wu</a><sup>2</sup>,
    <a href="http://www.cad.zju.edu.cn/home/gfzhang/">Guofeng Zhang</a><sup>3</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1✉️</sup>
    <br>
        <sup>1</sup>HKUST,
        <sup>2</sup>Monash University,
        <sup>3</sup>Zhejiang University
</p>

<div align="center">
    <a href=https://arxiv.org/abs/2404.13679><img src='https://img.shields.io/badge/arXiv-2404.13679-b31b1b.svg'></a>  
    <a href='https://w-ted.github.io/publications/gscream/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
</div>
</p>


## Installation

```
git clone https://github.com/W-Ted/GScream.git

cd GScream
conda env create -f gscream.yaml
conda activate gscream

cd submodules/diff-gaussian-rasterization/ && pip install -e .
cd ../simple-knn && pip install -e .
cd ../..
```
Since we used RTX 3090, in the [setup.py](https://github.com/W-Ted/GScream/blob/e7cc71bf3e878d02d15b524fdb44f038eba59a2a/submodules/diff-gaussian-rasterization/setup.py#L29), we hardcoded the gencode=arch with 'compute_86' and 'sm_86' when compiling 'diff-gaussian-rasterization'. For Tesla V100, you may try changing it to 'compute_70' and 'sm_70' before compiling. [issue#4](https://github.com/W-Ted/GScream/issues/4)

## Dataset

We provide the processed SPIN-NeRF dataset with Marigold depths [here(~9.7G)](https://drive.google.com/file/d/1EODx3392p1R7CaX5bazhkDrfrDtnqJXv/view?usp=sharing). You could download it to the ''data'' directory and unzip it. 

```
cd data
pip install gdown && gdown 'https://drive.google.com/uc?id=1EODx3392p1R7CaX5bazhkDrfrDtnqJXv'
unzip spinnerf_dataset_processed.zip && cd ..
```

Please refer to [SPIN-NeRF dataset](https://drive.google.com/drive/folders/1N7D4-6IutYD40v9lfXGSVbWrd47UdJEC) for the details of this dataset.  

## Training & Evaluation

```
python scripts/run.py
```
All the results will be save in the ''outputs'' directory. 

## Acknowledgements

This project is built upon [Scaffold-GS](https://city-super.github.io/scaffold-gs). The in-painted images are obtained by [SD-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) and [LaMa](https://github.com/advimman/lama). The depth maps are estimated by [Marigold](https://marigoldmonodepth.github.io/). The dataset we used is proposed by [SPIN-NeRF](https://spinnerf3d.github.io/). Kudos to these researchers. 

## Citation

```BibTeX
@inproceedings{wang2024gscream,
     title={GScream: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal},
     author={Wang, Yuxin and Wu, Qianyi and Zhang, Guofeng and Xu, Dan},
     booktitle={ECCV},
     year={2024}
     }
```
