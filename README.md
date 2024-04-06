# MPHY0012: PitSAM

## PitSAM: Low-Rank Adaptation of SAM for Pituitary Adenoma Segmentation

**PitSAM** represents a methodology for adapting the **Segment Anything Model (SAM)** to the task of segmenting pituitary adenomas from MRI scans. Employing the **Low-Rank Adaptation (LoRA)** technique, PitSAM fine-tunes the SAM model with a minimal increase in trainable parameters, facilitating an efficient transition to the medical imaging sphere.

### Overview

Accurate segmentation of pituitary adenomas from MRI scans is a cornerstone of effective clinical decision-making and treatment planning. Traditional machine learning strategies often fall short, challenged by the scarcity of labeled medical data and the distinctive properties of MRI scans. PitSAM surmounts these obstacles by tailoring the robust SAM model to the pituitary adenoma segmentation task. This adaptation preserves the foundational model's pre-trained knowledge and capabilities, ensuring a targeted and effective segmentation approach.

### Key Features

- **Gated Attention Mechanism:** A gated attention mechanism is integrated within the LoRA layers. 

- **Multi-View Learning:** Exploiting MRI data from all three anatomical planes (axial, coronal, and sagittal), PitSAM enriches its training dataset. 

