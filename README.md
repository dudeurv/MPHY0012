# MPHY0012

PitSAM: Low-Rank Adaptation of SAM for Pituitary Adenoma Segmentation
This repository contains the code for PitSAM (Pituitary Segmentation Adaptation Model), a novel approach for adapting the Segment Anything Model (SAM) to the task of pituitary adenoma segmentation from magnetic resonance imaging (MRI) scans. PitSAM leverages Low-Rank Adaptation (LoRA) techniques to fine-tune the SAM model with minimal additional trainable parameters, enabling efficient adaptation to the medical imaging domain.

Overview
Accurate segmentation of pituitary adenomas from MRI scans is crucial for clinical decision-making and treatment planning. However, the scarcity of labeled medical data and the unique characteristics of MRI data pose significant challenges for traditional machine learning approaches. PitSAM addresses these challenges by adapting the powerful Segment Anything Model (SAM) to the specific task of pituitary adenoma segmentation, while maintaining the pre-trained knowledge and capabilities of the foundation model.

Key features of PitSAM:

Low-Rank Adaptation (LoRA): PitSAM incorporates LoRA into the SAM model, enabling efficient fine-tuning with minimal additional trainable parameters.
Gated Attention: A novel gated attention mechanism is integrated into the LoRA layers, allowing PitSAM to capture domain-specific features and spatial information crucial for accurate pituitary adenoma segmentation.
Multi-View Learning: PitSAM leverages MRI data from all three anatomical planes (axial, coronal, and sagittal) during training, enriching the model's dataset and improving learning outcomes.
