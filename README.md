# Local Alignment Between Text and Image Features

## Problem Statement

Traditional CLIP models face an architectural mismatch:

* **Image encoders** output a fixed number of patch features (e.g., 49 patches for a 7 × 7 grid).
* **Text encoders** output a variable-length sequence of token embeddings.

Because the two modalities differ in length, it is hard to learn **fine-grained correspondences** between specific text semantics and specific image regions.

## Proposed Solution

We enhance CLIP with a **local-alignment mechanism** that learns spatial correspondences while preserving CLIP’s strong global alignment.

### 1. Text Feature Compression Module

A learnable compression module converts the variable-length text features into a fixed-size tensor whose length matches the number of image patches.

$$
\begin{aligned}
\mathbf{T} &= \text{TextEncoder}(\text) \in \mathbb{R}^{N \times d} \\[4pt]
\mathbf{I} &= \text{ImageEncoder}(\text{image}) \in \mathbb{R}^{M \times d} \\[4pt]
\mathbf{T}_{\text{compressed}} &= \text{CompressionModule}\!\bigl(\mathbf{T},\,\mathbf{R}\bigr) \in \mathbb{R}^{M \times d}
\end{aligned}
$$

where

* $N$ = number of tokens, $M$ = number of patches, $d$ = embedding dim
* $\mathbf{R}\in\mathbb{R}^{M\times d}$ are **learnable register tokens** that act as semantic “slots”
* $\mathbf{T}_{\text{compressed}}$ is aligned one-to-one with image patches

**Inside the module**

* Cross-attention lets each register pull the text information it needs.
* Registers are updated and returned as $\mathbf{T}_{\text{compressed}}$.
* Gradients flow through to the text encoder, so the entire model trains end-to-end.

### 2. Multi-Level Objective

The overall loss combines global CLIP contrastive loss and a new local alignment loss:

$$
\mathcal{L}_{\text{total}}
  = \lambda_{\text{global}}\,\mathcal{L}_{\text{global}}
  \;+\;
  \lambda_{\text{local}}\,\mathcal{L}_{\text{local}} .
$$

* **Global loss** $\mathcal{L}_{\text{global}}$: standard CLIP loss on pooled image/text features.
* **Local loss** $\mathcal{L}_{\text{local}}$: patch-wise alignment term:

$$
\mathcal{L}_{\text{local}}
  \;=\;
  \sum_{i=1}^{M}
    \operatorname{CrossEntropy}\!\bigl(
      \operatorname{sim}\,(
        \mathbf{T}_{\text{compressed}}^{(i)},
        \mathbf{I}^{(i)}
      )
    \bigr).
$$

### 3. Expected Benefits

1. **Fine-grained understanding** — learns phrase-to-region links.
2. **Improved localization** — boosts tasks that need spatial reasoning.
3. **Semantic grounding** — language concepts are explicitly tied to visual elements.
4. **No loss of global strength** — keeps CLIP’s original retrieval power.

### 4. Applications

* **Visual Question Answering**: richer spatial context for complex queries.
* **Image Captioning**: more precise object placement and relations.
* **Object Detection / Phrase-Grounding**: better text-to-box correspondence.
* **Fine-Grained Retrieval**: discriminate on subtle region-level details.
