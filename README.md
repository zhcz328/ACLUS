```

```

# A SAM-Guided Efficient Anatomy-Level Self-Supervised Pre-Training Method for Ultrasound Image Analysis

## Abstract

**The demand for efficient pre-training techniques to improve diagnostic efficiency and accuracy has grown as artificial intelligence in medical ultrasound (US) image analysis continues to progress. Prior pre-training methods for US images are either region-based or only concentrate on full images or video frames, failing to address the unique characteristics and capture the detailed anatomical features of ultrasound images. In this study, we propose \textbf{A}natomy-level \textbf{C}ontrastive \textbf{L}earning for \textbf{U}ltra\textbf{s}ound Images (ACLUS), a self-supervised pre-training method that focuses on anatomical features essential for medical interpretation. To bridge the domain gap between natural and ultrasound images, A-SAM is proposed and adapted utilizing using pre-existing image-mask pairs. This eliminates the need for human intervention and enables precise and automated segmentation. Building on this capability, ACLUS performs cross-view contrastive learning at the anatomical level, focusing on inter-anatomy relationships. Furthermore, considering the boundaries of ultrasound image irrelevance, ACLUS integrates context prediction within the damaged core region, interplaying with anatomy-level contrast to capture fine-grained anatomical details. Experimental results on multiple ultrasound datasets and diverse downstream tasks show that ACLUS substantially enhances pre-training quality in the ultrasound domain, consistently surpassing state-of-the-art methods while offering a computationally efficient framework well-suited for practical clinical deployment.**

![ACLUS](./figs/ACLUS.png)

## 🔨 PostScript

😄 This project is the pytorch implemention of ACLUS

😆 Our experimental platform is configured with two *RTX3090* GPUs

## 💻 Installation

1. Clone or download this repository.

   ```
   cd <ACLUS_project_dir>
   ```

2. Create conda environment.

   ```
   conda create -n ACLUS python=3.9
   conda activatee ACLUS
   ```

3. Install dependencies.

   ```
   pip install -r requirements.txt
   ```

## 🐾 ACLUS Evaluation

1. Download our pre-trained ACLUS weights [pre-trained-aclus.pth](https://drive.google.com/file/d/1n8A3vK2UGE7g_7keC_NBfAa1HuKHp-Pz/view?usp=sharing)  

2. Download the 5 fold cross validation [POCUS](https://drive.google.com/file/d/1w7FrwqQ09VjwtTcZL5M0hZnW3Oly9Buv/view?usp=drive_link) dataset.

3. Run the demo with:

   ```
   python ./evaluation_aclus.py --data_path <pocus_data_path> --ckpt_path <pre_trained_aclus_path>
   ```

## 📘 Finetune SAM

### Checkpoint

Download checkpoint for SAM (Segment Anything Model): [ViT_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### Dataset

1. Download datasets, including [DDTI]( https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [TG3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [CAMUS](http://camus.creatis.insa-lyon.fr/challenge/) , and then convert these datasets into `.png` format.

2. Then these datasets should be set in the format of `./SAM/dataset/` folder.

   ```none
   dataset
   ├── ThyroidGland-DDTI
   │   ├── img
   │   │   ├── xxx.png
   │   │   ├── ... 
   │   ├── lable
   │   │   ├── xxx.png
   │   │   ├── ...
   │── Echocardiography-CAMUS
   │   ├── img
   │   │   ├── xxx.png
   │   │   ├── ... 
   │   ├── lable
   │   │   ├── xxx.png
   │   │   ├── ...
   .........
   │── MainPatient
   ```

3. The `./SAM/MainPatient` folder contains the train/val.txt which has formatted line as:

   ```
     <class ID>/<dataset file folder name>/<image file name>
   ```

4. Set other configs in `./SAM/utils/config.py`.

### Finetune 

1. Finetune the SAM for the ultrasound domain with：

   ```
   python ./SAM/only_train_sam.py --sam_ckpt <sam_vit_b.pth> 
   ```

2. Finetune auto prompter with：

   ```
   python ./SAM/train_auto_prompter.py --sam_ckpt <finetuned_sam.pth> --load_auto_prompter
   ```

## 🐾 ACLUS Pre-training

1. > Download the datasets from [Butterfly](https://drive.google.com/file/d/1zefZInevopumI-VdX6r7Bj-6pj_WILrr/view?usp=sharing) and [HMC-QU](https://aistudio.baidu.com/aistudio/datasetdetail/102406) and place them in a formatted directory after generating masks through A-SAM inference as shown below:
   >
   > ```
   > dataset
   > ├── train
   > │   ├──Butterfly
   > │   │   ├──img
   > │   │   ├──label
   > │   ├──HMC-QU
   > │   │   ├──img
   > │   │   ├──label
   > ```

2. Run the `./utils/generate_masks_pkls.py` to generate masks .pkl files in './dataset/masks' 
   and indexes at `./dataset/train_tf_img_to_gt.pkl`.

   ```
   dataset
   ├── train
   │   ├──Butterfly
   │   │   ├──img
   │   │   ├──label
   │   ├──HMC-QU
   │   │   ├──img
   │   │   ├──label
   │── masks
   │   ├──xxx.pkl
   │── train_tf_img_to_gt.pkl
   ```

3. Set training options in config file: `./configs/example.yaml`, and the `resume_path` is need to be assigned 
   to the checkpoint file you want to resume training. 

4. Train the ACLUS model with:

   ```
   python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 pretrain_aclus.py --cfg ./configs/example.yaml
   ```
