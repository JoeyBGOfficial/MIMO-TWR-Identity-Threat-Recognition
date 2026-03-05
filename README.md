![MIMO](https://github.com/user-attachments/assets/2d7dee34-9bb9-4058-8f55-a09276574534)

# MIMO TWR Identity Threat Recognition

<p align="center">
  <a href="https://www.semanticscholar.org/author/Weicheng-Gao/2051685234"><img src="https://img.shields.io/badge/Semantic_Scholar-464EB8?style=for-the-badge&logo=semanticscholar&logoColor=white" alt="Semantic Scholar"/></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://joeybgofficial.github.io/"><img src="https://img.shields.io/badge/Personal_Homepage-252525?style=for-the-badge&logo=github&logoColor=white" alt="Personal Homepage"/></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ieeexplore.ieee.org/author/37089574449"><img src="https://img.shields.io/badge/IEEE-00629B?style=for-the-badge&logo=ieee&logoColor=white" alt="IEEE"/></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://radar.bit.edu.cn/index.htm"><img src="https://img.shields.io/badge/Team_Website-005A3C?style=for-the-badge&logo=rss&logoColor=white" alt="Team Website"/></a>
</p>

## I. INTRODUCTION

### Write Sth. Upfront

**Through-the-Wall Radar (TWR) Human Activity Recognition (HAR) represents a cutting-edge field in pattern recognition research.** By extracting micro-Doppler signature of indoor targets under conditions of penetration, low signal-to-noise ratio, low resolution, and multipath effects, this field enables precise analysis of human motion states in completely sheltered spaces, providing robust technical support for urban security surveillance. However, existing works have only focused on classifying different activities exhibited by the same individual, without exploring more refined topics such as personnel identification and threat identification, thus limiting its practical value. **This work is the first known effort to achieve simultaneous recognition of human identity and threat assessment in completely sheltered spaces using TWR.** We hope this will provide a groundbreaking, high-quality, and systematic research for our peers!

**We fully trust our peer community and welcome the use of our open-source code for one-click verification, ensuring the reproducibility of the reported results in our paper.**
\
\
![1](https://github.com/user-attachments/assets/474711ee-f0cf-4139-8bb6-52cd8f32a456)

![2](https://github.com/user-attachments/assets/af4d5d0c-ccd4-4ef4-bcb2-e835d0dc0735)

### Basic Information:

This repository is the open source code for my latest work: "MIMO Through-the-Wall Radar Identity-Threat Recognition via Multi-Channel Fusion and Riemannian Micro-Doppler Representation", submitted to IEEE TPAMI.

**My Email:** JoeyBG@126.com;

**Abstract:** Prior information regarding friendly and hostile forces in urban warfare and counter-terrorism operations is provided by multiple-input multiple-output (MIMO) through-the-wall radar (TWR), achieving non-line-of-sight sensing of identity and threat. However, the distinctiveness of micro-Doppler signature between armed and unarmed walking human is minimized by the low signal-to-noise ratio, resolution, and limited training data of TWR, causing the recognition of existing methods ineffective. To address this issue, a MIMO TWR identity-threat recognition method based on multi-channel fusion and Riemannian micro-Doppler representation is proposed in this paper. The kinematics model for armed and unarmed human and TWR echo model are first established. Then, a fusion method based on entropy minimization and peak signal-to-noise ratio (PSNR) screening is proposed, alongside a fusion method based on trace ratio group sparse (TRGS) feature selection. The augmentation of multi-channel micro-Doppler signature is achieved. Next, a feature representation method based on Riemannian manifold geometry is proposed. The edge and detail information of micro-Doppler signature on radar images is enhanced. Finally, a neural network model based on multi-stream deep ensemble is proposed, and the integrated recognition of human identity and threat for TWR is achieved. Numerical simulations and experiments are conducted to verify the effectiveness of the proposed method, which demonstrates good accuracy under multiple individuals regardless of whether they are armed.

**Corresponding Papers:**

[1] (Upon Reviewing...)

## II. HOW TO REPRODUCE

All source code in the repository is well-structured, extensively commented, and thoroughly debugged. With MATLAB R2025a or later installed, the scripts are designed to be executed in one click.

The 'SimH_Set' folder contains the complete pipeline for simulation modeling, data processing, feature extraction, and recognition. The 'RW_Set' folder provides the full workflow for real-world data processing, feature extraction, and recognition. Additionally, the 'Benchmark' folder includes the comprehensive implementation and reproduction of the five state-of-the-art methods compared in this study.

**⭐By following the steps outlined below, the exact simulated results presented in this paper can be immediately reproduced:⭐**

(1) Download the whole repository and unzip. Add the entire repository to the MATLAB search path.

(2) Enter 'SimH_Set' folder.

(3) Open 'Home' -> 'Add-Ons' -> 'Explore Add-Ons'.
\
\
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ee969ae2-58aa-4463-81c1-99bc807e6c4b" />
\
\
(4) Search 'Deep Learning ToolboxTM Model for Xception Network' and install the model.
\
\
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0fc51358-485c-4917-905e-4eafbfba9304" />
\
\
(5) After the installing is completed, run the following scripts in sequence. Please follow the sequence below strictly! Each script should only be run after the previous one has finished executing: "SimH_Dataset_Generator_PSNR.m" -> "SimH_Dataset_Generator_TRGS.m" -> "SimH_Dataset_Merging.m" -> "SimH_Riemann_Featureset_Generator.m". Once the execution is complete, your MATLAB workspace should appear as shown below:
\
\
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f9892a47-ec19-4300-a841-250a8c54f083" />
\
\
(6) Run "SimH_Recognition_Model.m". When all the models are trained and validated, the figures will be generated automatically.

**⭐By following the steps outlined below, the exact measured results presented in this paper can be immediately reproduced:⭐**

(1) Download the whole repository and unzip. Add the entire repository to the MATLAB search path.

(2) Enter 'RW_Set' folder.

(3) Download all the image-based dataset from this link: https://drive.google.com/file/d/11STB1Kyb1r5GYCp3O-R3QM9s9NHmCci1/view?usp=sharing. Unzip the downloaded RW_Set.rar file to the current folder. There should be four image dataset folders in 'RW_Set' path: RW_RTM_Set, RW_DTM_Set, RW_RDM_Set, and RW_Feature_Set. Make sure your MATLAB workspace appear as shown below:
\
\
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/00e707cd-6474-49e3-9478-204664180ff1" />
\
\
(4) Open 'Home' -> 'Add-Ons' -> 'Explore Add-Ons'.
\
\
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ee969ae2-58aa-4463-81c1-99bc807e6c4b" />
\
\
(5) Search 'Deep Learning ToolboxTM Model for Xception Network' and install the model.
\
\
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0fc51358-485c-4917-905e-4eafbfba9304" />
\
\
(6) After the installing is completed, run "RW_Recognition_Model.m". When all the models are trained and validated, the figures will be generated automatically.

## III. SOME THINGS TO NOTE

**(1) Environment Issues:** The project consists of pure MATLAB code. The recommended MATLAB version is R2025a and above. The program is executed by the CPU environment, but GPU version is also provided. The key innovation of this paper, including all the signal processing, feature augmentation, and feature representation scripts are written by me. The network model of this paper is written by Gemini 3 Pro and debugged by me. If you have any question about the reproduction, feel free to email me at any time: JoeyBG@126.com.

**(3) Algorithm Design Issues:** This work addresses a pioneering scientific problem. By benchmarking against several state-of-the-art methods reproduced from other sub-fileds on TWR HAR, we demonstrate that our proposed approach consistently outperforms existing methods by over 10% in recognition accuracy.

**(4) Right Issues: ⭐Considering intellectual property and the hard work of many team members, this work only open-sources my code and visualization results, not the real-world raw data. Additionally, this project is strictly for learning purposes only. Any direct use for paper submissions, patents, or commercialization must receive our consent!⭐**
