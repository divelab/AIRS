# GeoMPNN

## Introduction

This is the official implementation of GeoMPNN, winner of the best Student Submission award at the NeurIPS 2024 ML4CFD Competition, placing fourth overall.  

**Team**: Jacob Helwig, Xuan Zhang, Haiyang Yu, and Shuiwang Ji

## Environment Setup

```
source setup.sh
```

## Data Download

```
python download_data.py
```

## Create Session

```
python create_session.py
```

## Train

```
conda activate GeoMPNN
gpu=8
CUDA_VISIBLE_DEVICES=$gpu python ingestion.py -1 GeoMPNN/ ./ GeoMPNN/
```

## Acknowledgments

This work was supported in part by National Science Foundation grants IIS-2243850 and CNS-2328395.
