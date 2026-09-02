# Real-Time Visual Clutter & Risk Assessment of Electrical Posts Using Deep Learning V2
an updated version of the project I made for our Deep Learning subject back in uni called, Real-Time Visual Clutter and Risk Assessment of Electrical Posts Using Deep Learning. 

In here, we will be making use of Instance Segmentation with polygon masks in order to identify the exact pixels belonging to the wire clusters and those that belongs to the pole, efficiently distinguishing them from its background interferences.

While the two-model pipeline (YOLO for cropping, EfficientNet for classification) have provided sufficient results, it still lack the ability to classify the wire clusters from, say, trees or sky---resulting to false classification. An end-to-end segmentation will address this loopholes using YOLO instance segmentation.

Also, for the severity logic, we will be making use of a deterministic score instead of predicting out from the cropped box. In order to do this, the deterministic score will calculate the mask pixels of the wire clusters from the mask pixels of the pole. The severity range will then depend on the resulting score, that is, whether it's a relatively safe electrical posts.

This time, I will attempt to have a larger dataset than my previous 500 something images.

**by:** Kyla Reambonanza

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
[web:25][web:41]

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Electrical posts in several urban areas of third-world countries like the Philippines, often contains clusters of tangled, overloaded, or poorly maintained wiring, which are commonly known as "spaghetti wires". These wires can often be hazardous which can pose risks of electrical fires, electrocution, and service interruptions. Manual inspection by local authorities can be time-consuming, inconsistent, and limited to manpower. 

Therefore, this project aims to develop a deep learning- based system that analysises images of electrical posts and classifying them into multiple risk levels. By leveraging computer vision, the system can offer safety monitoring, early hazard detection, improve maintenance planning, and overall help authorities prioritise which poles to clean up first. In Philippine context, this project satifies Anti-Obstruction of Power Lines Act or the R.A. 11361, which ensures the smooth and uninterrupted transmission of electricity.

### Objectives
- Collect newer and better dataset
- Replace bounding boxes with Instance Segmentation using polygon masks
- Develop a unified end-to-end multitask architecture

![Problem Demo](images/problem_example.gif)[web:41]

## Methodology
### Dataset
- Source: Custom Dataset of Electrical Posts collected via web scraping, open-source repositories, Google Maps, Kartaview, and original photography of urban electrical posts

- Size: Approximately 600-800 images 

- Split: 70% Train, 15% Validation, 15% Test.
    * Train: ~ -> ~ (Augmented)
    * Validation: ~ images
    * Test: ~ images

- Preprocessing: 

### Architecture
- Head: Custom Classification Head
- Hyperparameters: Table below

| Parameter | YOLOv8n |
|-----------|-------| 
| Batch Size | - | 
| Learning Rate | - | 
| Epochs | 50 (stopped at -) |
| Optimizer | AdamW | 

### Training Code Snippet
**YOLOv8 Detection Training**


## Experiments & Results
### Metrics
| Model | mAP@0.5 | Precision | Recall | Inference Time (ms) |
|-------|---------|-----------|--------|---------------------|
| Baseline (YOLOv8n) | - | - | - | - |
| **Ours (YOLO-detector)** | **-** | **-** | **-** | **-** |



### Demo
Video Demo:[web:41]

## Discussion
- 
    

## Ethical Considerations
- Bias: Dataset skew toward specific types of wires/poles
- Privacy: No human subjects or identifying information is present in the training data.
- Misuse: Potential application for automated infrastructure inspection, which, if repurposed without consent, could raise surveillance or property rights issues [web:41]

## Conclusion
-

## Installation
1. Clone repo: `git clone https://github.com/kay-16/electrical-post-clutter-segmentation-V2.git`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: Run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
- torch>=2.0
- ultralytics
- opencv-python
- albumentations


## Related Work
- <div class="csl-entry">Kim, J., Kamari, M., Lee, S., &#38; Ham, Y. (2021). Large-Scale Visual Data–Driven Probabilistic Risk Assessment of Utility Poles Regarding the Vulnerability of Power Distribution Infrastructure Systems. <i>Journal of Construction Engineering and Management-Asce</i>, <i>147</i>(10), 04021121. https://doi.org/10.1061/(ASCE)CO.1943-7862.0002153</div>

- <div class="csl-entry">Benelmostafa, B.-E., &#38; Medromi, H. (2025). PowerLine-MTYOLO: A Multitask YOLO Model for Simultaneous Cable Segmentation and Broken Strand Detection. <i>Drones</i>, <i>9</i>(7), 505. https://doi.org/10.3390/drones9070505</div>

- Gap addressed:  
    * The Two-Stage Risk Assessment Pipeline - While other works use two stages to find a component and then classify its defect type, my pipeline is designed for abstract risk assessment using EfficientNetB0 (for classification) and YOLOv8n (for detection)

    * The dataset and target - By detecting formless, tangled wire clusters rather than distinct, standardized hardware components (like insulators, dampeners, or towers)
[web:25]



## References
[1] <div class="csl-entry">Kim, J., Kamari, M., Lee, S., &#38; Ham, Y. (2021). Large-Scale Visual Data–Driven Probabilistic Risk Assessment of Utility Poles Regarding the Vulnerability of Power Distribution Infrastructure Systems. <i>Journal of Construction Engineering and Management-Asce</i>, <i>147</i>(10), 04021121. https://doi.org/10.1061/(ASCE)CO.1943-7862.0002153</div>

[2] <div class="csl-entry">Benelmostafa, B.-E., &#38; Medromi, H. (2025). PowerLine-MTYOLO: A Multitask YOLO Model for Simultaneous Cable Segmentation and Broken Strand Detection. <i>Drones</i>, <i>9</i>(7), 505. https://doi.org/10.3390/drones9070505</div> [web:25]

## GitHub Pages
View project site at: https://kay-16.github.io/electrical-post-clutter-segmentation-V2/ [web:32]