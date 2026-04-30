---
paper_id: focalloss-lin2017
citation_key: FocalLoss-lin2017
title: "Focal Loss for Dense Object Detection"
year: 2017
arxiv_id: 1708.02002
doi: 10.1109/TPAMI.2018.2858826
url: https://arxiv.org/abs/1708.02002
semantic_scholar_paper_id: 1a857da1a8ce47b2aa185b91b5cb215ddef24de7
semantic_scholar_citation_count: 3215
source_kind: Bib
download_mode: MetadataOnly
has_local_tex: false
has_local_pdf: false
parse_status: MetadataOnly
kg_tags: []
---

# Focal Loss for Dense Object Detection

## Metadata

- Citation key: FocalLoss-lin2017
- Year: 2017
- arXiv: 1708.02002
- DOI: 10.1109/TPAMI.2018.2858826
- URL: https://arxiv.org/abs/1708.02002

## Semantic Scholar

- Paper ID: 1a857da1a8ce47b2aa185b91b5cb215ddef24de7
- Corpus ID: 206771220
- Citation count: 3215
- Influential citation count: 206
- Fields of study: Medicine, Computer Science
- TLDR: This paper proposes to address the extreme foreground-background class imbalance encountered during training of dense detectors by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples, and develops a novel Focal Loss, which focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training.

## Abstract

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. Code is at: https://github.com/facebookresearch/Detectron.

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