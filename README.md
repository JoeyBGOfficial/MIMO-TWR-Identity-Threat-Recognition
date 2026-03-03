# MIMO-TWR-Identity-Threat-Recognition

## I. INTRODUCTION

### Write Sth. Upfront

This work stands apart from my through-the-wall radar (TWR) human activity recognition (HAR) series and represents an attempt to employ relatively independent classical signal processing techniques.

I don’t consider this work to be particularly forward-looking or exceptionally effective. However, existing radar-based HAR studies, whether through-the-wall or in free space, typically rely on single-channel echoes to generate range or Doppler profiles. Even in the few cases where multi-input-multi-output (MIMO) radar is used, the focus is generally on generating angle spectra or imaging sequences. In fact, these two ideas can be combined by leveraging Doppler feature from different MIMO radar channels, performing fusion, compensation, and feature transformation into a single channel to achieve the goal of enhancing micro-Doppler signature. Based on the above considerations, this work came into being.

Although this work does not involve neural networks, I believe it is still worth sharing openly. I also hope that open-sourcing it can advance the field of radar micro-Doppler signature extraction and HAR.

### Basic Information:

This repository is the open source code for my latest work: "MIMO Through-the-Wall Radar Micro-Doppler Signature Augmentation Method Based on Multi-Channel Information Fusion", submitted to IEEE Signal Processing Letters.

**My Email:** JoeyBG@126.com;

**Abstract:** TWR can monitor and analyze the motion characteristics and activity patterns of indoor human targets, with the advantages of non-contact, high flexibility and privacy protection. However, existing TWR HAR techniques developed based on single-channel radar contain limited Doppler information, making it difficult to achieve accurate recognition on data where the direction of human motion is not parallel to the radar observation. To solve this problem, in this letter, a MIMO TWR micro-Doppler signature augmentation method based on multi-channel information fusion is proposed. First, a multi-channel Doppler profile feature fusion method based on multi-scale wavelets with low-rank decomposition is presented. Then, a motion parameter estimation method based on Broyden-Fletcher-Goldfarb-Shanno (BFGS) global optimization is proposed, and the fused Doppler profile transformation is implemented using the obtained orientation of human motion. Numerical simulated and measured experiments demonstrate the effectiveness of the proposed method.

**Corresponding Papers:**

[1] W. Gao, S. Liu, J. Wang, X. Qu and X. Yang, "MIMO Through-the-Wall Radar Micro- Doppler Signature Augmentation Method Based on Multi-Channel Information Fusion," in IEEE Signal Processing Letters, 2025. Link: https://ieeexplore.ieee.org/document/11345966.
