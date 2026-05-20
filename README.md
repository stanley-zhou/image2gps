# Image2GPS: Campus-Scale Geolocation with Visual Retrieval

This repository contains the final notebook and report for an Image2GPS project in CIS 4190/5190 Applied Machine Learning. The goal is to predict the GPS coordinates of a photo taken around the Penn Engineering campus region using computer vision models.

## Project Overview

Image2GPS is a fine-grained image geolocation task. Given an input campus image, the model predicts the latitude and longitude of the camera location. Performance is evaluated using average Haversine distance in meters, where lower is better.

The project began with supervised coordinate regression models and gradually evolved into a retrieval-based localization system. In a finite campus environment with repeated walkway viewpoints, retrieving visually similar training images and averaging their GPS coordinates worked better than directly regressing latitude and longitude from pixels.

## Final Approach

The final submitted model uses a frozen DINOv2 ViT-L/14 encoder and a temperature-weighted k-nearest-neighbor retrieval strategy:

1. Encode training images into normalized DINOv2 embeddings.
2. Store the training embedding gallery together with raw GPS coordinates.
3. At inference time, encode the query image using test-time augmentation.
4. Retrieve the most visually similar training images using cosine similarity.
5. Predict GPS coordinates using a softmax-temperature-weighted average of the top-k retrieved coordinates.

This non-parametric retrieval approach requires no gradient updates to the DINOv2 encoder and directly leverages self-supervised visual representations for fine-grained place recognition.

## Key Results

| Model / Stage | Average Haversine Error |
|---|---:|
| Released ResNet-18 baseline | 90.4 m |
| Best supervised regression model | ~27.7 m |
| Strong DINOv2 retrieval variants | ~22-24 m internal validation |
| Best internal ensemble | ~19.9 m internal validation |
| Final public leaderboard submission | 35.48 m |

Although spatial priors, stable-cue retrieval, and ensembling achieved the best internal validation results, the final submission prioritized generalization and used the simpler DINOv2-L retrieval model with test-time augmentation.

## Repository Contents

```text
.
├── README.md
├── Image2GPS_Notebook_Final.ipynb
└── Image2GPS_Report_Final.pdf
```

- `Image2GPS_Notebook_Final.ipynb`: final experimental notebook, including data preparation, model development, retrieval methods, post-processing, ensembling, evaluation, and diagnostic analysis.
- `Image2GPS_Report_Final.pdf`: final written report summarizing the task, dataset, methods, experiments, results, exploratory analysis, and limitations.

## Dataset and External Resources

- Hugging Face dataset: https://huggingface.co/datasets/stanley-zhou/Image2GPS
- Final notebook backup: https://drive.google.com/file/d/16MGv2uClWdgemNM3GcVZV-xKJ6qpfPzg/view?usp=sharing
- Demo video folder: https://drive.google.com/drive/folders/1InQw5Mk3Uav6i2Z5tz8YUKHU2WOSymm3?usp=sharing

## Main Techniques Explored

- ResNet-18 supervised GPS regression baseline
- ImageNet-pretrained regression models
- Geolocation-aware image augmentation
- EfficientNet-B0 regression with an MLP head
- DINOv2 ViT-L/14 embedding extraction
- Cosine-similarity k-NN retrieval
- Temperature-weighted coordinate averaging
- Test-time augmentation for retrieval robustness
- Grid-guided retrieval and walkway-prior snapping
- Stable-cue multi-crop retrieval
- Fixed-weight and learned ensembling
- Worst-case retrieval and spatial error analysis

## Main Takeaways

The central finding is that constrained-environment geolocation is better treated as fine-grained visual localization than as unconstrained coordinate regression. Supervised regression improved substantially over the released baseline but plateaued because ambiguous images tended to be predicted near an average campus location. In contrast, DINOv2 retrieval preserved local visual distinctiveness and produced more accurate predictions by matching each query image to visually similar known locations.

The largest remaining errors occurred when visually similar campus structures appeared in different locations or when retrieved neighbors were geographically dispersed. Future improvements should focus on uncertainty-aware retrieval, targeted data collection in high-error regions, and confidence-adaptive spatial constraints.
