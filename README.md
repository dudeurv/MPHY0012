# MPHY0012: PitSAM

# PitSAM: Low-Rank Adaptation of SAM for Pituitary Adenoma Segmentation

This repository contains the code for PitSAM (Pituitary Segmentation Adaptation Model), a novel approach for adapting the Segment Anything Model (SAM) to the task of pituitary adenoma segmentation from magnetic resonance imaging (MRI) scans. PitSAM leverages Low-Rank Adaptation (LoRA) techniques to fine-tune the SAM model with minimal additional trainable parameters, enabling efficient adaptation to the medical imaging domain.

## Overview

Accurate segmentation of pituitary adenomas from MRI scans is crucial for clinical decision-making and treatment planning. However, the scarcity of labeled medical data and the unique characteristics of MRI data pose significant challenges for traditional machine learning approaches. PitSAM addresses these challenges by adapting the powerful Segment Anything Model (SAM) to the specific task of pituitary adenoma segmentation, while maintaining the pre-trained knowledge and capabilities of the foundation model.

Key features of PitSAM:

1. **Low-Rank Adaptation (LoRA)**: PitSAM incorporates LoRA into the SAM model, enabling efficient fine-tuning with minimal additional trainable parameters.
2. **Gated Attention**: A gated attention mechanism is integrated into the LoRA layers, allowing PitSAM to capture domain-specific features and spatial information crucial for accurate pituitary adenoma segmentation.
3. **Multi-View Learning**: PitSAM leverages MRI data from all three anatomical planes (axial, coronal, and sagittal) during training, enriching the model's dataset and improving learning outcomes.

## Installation

1. Clone this repository:

```
git clone https://github.com/dudeurv/PitSAM_MPHY0012.git
```

2. Install the required dependencies by running the provided script:

```
bash sequential_install.sh
```

3. Download the pre-trained Segment Anything Model (SAM) weights from the official repository: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

## Usage

1. Prepare your dataset following the required format.
2. Run the training script:


```
python training_with_inference.py --data_path /path/to/your/data/folder --output_file /path/to/your/output/file.pth
```

This command will start the training process using the data from the specified `--data_path` and save the trained model checkpoint to the location specified by `--output_file`.

You can also include other arguments to customize the training process. For example:

```
python training_with_inference.py --data_path /path/to/your/data/folder --output_file /path/to/your/output/file.pth --batch_size 8 --max_epochs 100 --vit_name vit_l --rank 8
```

This command will use a batch size of 8, train for a maximum of 100 epochs, use the ViT-L model variant, and set the LoRA rank to 8.

Make sure to replace `/path/to/your/data/folder` and `/path/to/your/output/file.pth` with the appropriate paths on your system.

Here's a brief explanation of the arguments used in the example:

- `--data_path`: The path to the directory containing your training data.
- `--output_file`: The path and filename for the trained model checkpoint.
- `--batch_size`: The batch size for training (default is 12).
- `--max_epochs`: The maximum number of epochs to train for (default is 80).
- `--vit_name`: The name of the ViT model to use (default is `'vit_b'`).
- `--rank`: The rank for LoRA adaptation (default is 4).



The main components of the repository are:

- `dataloader.py`: Contains the custom data loader for loading and preprocessing the MRI data.
- `sam_lora_image_encoder.py`: Defines the PitSAM architecture, incorporating the gated attention mechanism into the LoRA layers of the SAM image encoder.
- `sam_lora_image_encoder_mask_decoder.py`: Combines the PitSAM image encoder with the mask decoder from SAM.
- `utils.py`: Utility functions for data processing, visualization, and evaluation.


## Results

PitSAM achieved state-of-the-art performance on our private pituitary adenoma segmentation dataset, with a Dice Similarity Coefficient (DSC) of 98.8% for tumor segmentation, 87.9% for internal carotid artery segmentation, and an overall mean DSC of 93.4%. These results were obtained with only 4.146 million trainable parameters, which is just 4.54% of the model's total parameters.

## Contributing

We welcome contributions to improve PitSAM and extend its capabilities. Please feel free to submit issues or pull requests to this repository.

## Acknowledgments

We would like to thank the National Hospital for Neurology and Neurosurgery for providing the pituitary adenoma MRI dataset used in this study.
