# Week 6 - Contrastive Learning & Vision-Language Models (study notes)

Written deliverable for the multimodal half of the project. Covers the theory
behind joining images and text, and how it feeds the report-writing stage.

## 1. The goal: a joint embedding space
An *embedding* is a vector (a list of numbers) representing something's meaning.
A **joint** embedding space maps **both images and text into the same vector
space**, arranged so that a picture of a scratch and the words "a scratch" land
near each other. Once you have this, comparing an image to text is just a
dot-product (nearby = similar). This is the foundation every modern
vision-language model is built on.

## 2. Contrastive learning (how you train that space)
The training rule is simple: **pull matching pairs together, push mismatched
pairs apart.**
- A *positive pair* = an image with its correct caption.
- *Negative pairs* = that image with every other caption in the batch.
The loss (**InfoNCE**) rewards high similarity for the positive pair and low
similarity for all the negatives. Do this over millions of pairs and the two
encoders (one for images, one for text) learn to place matching content nearby.

## 3. CLIP
CLIP (Radford et al., 2021) did exactly this at scale: ~400M image-text pairs
scraped from the web, an image encoder and a text encoder trained together.
For a batch of N pairs it builds an NxN similarity matrix; the N correct pairs
sit on the diagonal and are pushed up, the off-diagonal mismatches pushed down.

**Payoff - zero-shot classification:** give CLIP new class names as text
("crazing", "scratch", "inclusion"), embed them, embed an image, and pick the
nearest text. No retraining needed. That generality is why CLIP is everywhere.

## 4. BLIP / BLIP-2
BLIP extends CLIP by adding **generation** (captioning, Q&A), not just matching.
BLIP-2's trick is a lightweight "Q-Former" bridging a *frozen* image encoder to
a *frozen* language model cheaply. This lineage - contrastive alignment + a
bridge to an LLM - is essentially the recipe that produces modern VLMs (LLaVA,
GPT-4V), which is what writes the report in Week 7.

## 5. Why this matters for defect detection
Two concrete angles for the project:
- **Better features:** contrastive/self-supervised pretraining (CLIP, DINO, MoCo)
  learns richer visual representations than features learned from only 1,529
  labelled steel images. A CLIP-pretrained backbone could plausibly lift the
  weak classes (e.g. crazing) beyond what supervised YOLO alone achieves.
- **The report stage is multimodal by design:** the VLM consumes the raw image
  *and* the YOLO detections jointly (feature-level fusion). It is NOT the lazy
  pipeline of captioning the image to text first - the pixels reach the model
  directly, so texture and severity aren't lost.

## 6. One-line summary
*Contrastive learning aligns images and text in a shared space (CLIP); adding a
bridge to a language model turns that alignment into a VLM that can look at a
defect and write about it - which is exactly the final stage of this pipeline.*
