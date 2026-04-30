---
paper_id: groupnorm-wu2018
citation_key: GroupNorm-wu2018
title: "Group Normalization"
year: 2018
arxiv_id: 1803.08494
doi: 10.1007/s11263-019-01198-w
url: https://arxiv.org/abs/1803.08494
semantic_scholar_paper_id: d08b35243edc5be07387a9ed218070b31e502901
semantic_scholar_citation_count: 4293
source_kind: Bib
download_mode: MetadataOnly
has_local_tex: false
has_local_pdf: false
parse_status: MetadataOnly
kg_tags: []
---

# Group Normalization

## Metadata

- Citation key: GroupNorm-wu2018
- Year: 2018
- arXiv: 1803.08494
- DOI: 10.1007/s11263-019-01198-w
- URL: https://arxiv.org/abs/1803.08494

## Semantic Scholar

- Paper ID: d08b35243edc5be07387a9ed218070b31e502901
- Corpus ID: 4076251
- Citation count: 4293
- Influential citation count: 224
- Fields of study: Computer Science
- TLDR: Group Normalization (GN) is presented as a simple alternative to BN that can outperform its BN-based counterparts for object detection and segmentation in COCO, and for video classification in Kinetics, showing that GN can effectively replace the powerful BN in a variety of tasks.

## Abstract

Batch Normalization (BN) is a milestone technique in the development of deep learning, enabling various networks to train. However, normalizing along the batch dimension introduces problems—BN’s error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN’s usage for training larger models and transferring features to computer vision tasks including detection, segmentation, and video, which require small batches constrained by memory consumption. In this paper, we present Group Normalization (GN) as a simple alternative to BN. GN divides the channels into groups and computes within each group the mean and variance for normalization. GN’s computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes. On ResNet-50 trained in ImageNet, GN has 10.6% lower error than its BN counterpart when using a batch size of 2; when using typical batch sizes, GN is comparably good with BN and outperforms other normalization variants. Moreover, GN can be naturally transferred from pre-training to fine-tuning. GN can outperform its BN-based counterparts for object detection and segmentation in COCO (https://github.com/facebookresearch/Detectron/blob/master/projects/GN), and for video classification in Kinetics, showing that GN can effectively replace the powerful BN in a variety of tasks. GN can be easily implemented by a few lines of code in modern libraries.

## Section Map

- No structured sections were extracted.

## Main Sections

No local TeX source was available, so this paper is represented as metadata only.
## Figures And Tables

No figure or table captions were extracted.

## Citations

- No citation keys were extracted.

## Repo Relevance

- No repo relevance tags matched the current config.