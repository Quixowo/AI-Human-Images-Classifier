# AI vs. Real Image Classifier
> A lightweight, high-performance binary classifier using MobileNetV3 Transfer Learning to differentiate AI and real images.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Model](https://img.shields.io/badge/Model-MobileNetV3-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project Overview
With the rapid rise of Generative Adversarial Networks (GANs) and Diffusion models (like Midjourney, DALL-E, and Stable Diffusion), distinguishing between authentic photography and synthetic media has become a critical challenge for information integrity.

This project addresses this problem by building a **binary classification model** capable of detecting AI-generated artifacts. By leveraging **Transfer Learning** on the **MobileNetV3** architecture, the solution achieves high accuracy while remaining lightweight enough for edge deployment (e.g., mobile apps or browser plugins).

## Key Features
* **Transfer Learning:** Utilizes pre-trained weights from ImageNet to accelerate convergence.
* **Lightweight Architecture:** MobileNetV3Small/Large ensures low latency inference.
* **Data Augmentation:** Robust preprocessing pipeline (rotations, zooms, flips) to prevent overfitting.

## Tech Stack
* **Core:** Python, NumPy, Pandas
* **Deep Learning:** TensorFlow/Keras
* **Base Model:** MobileNetV3 (Pre-trained on ImageNet)
* **Visualization:** Matplotlib
* **Environment:** Google Colab

## Dataset
The model was trained on the **[AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)** sourced from Kaggle.
* **Total Images:** 80,000 (60,000 training and 20,000 validation)
* **Split:** 50% Real / 50% AI-Generated
* **Resolution:** Resized to 224x224 for MobileNetV3 input.

## Results
The model achieves exceptional performance, making it suitable for real-world screening applications.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.5%** |
| **Precision** | 99.3% |
| **Recall** | 99.4% |
| **F1-Score** | 99.4% |

### Confusion Matrix
<p align="left">
  <img src="https://github.com/user-attachments/assets/90f46939-ac3a-46b5-b695-9d0749d33009" width="500" title="Confusion Matrix">
</p>
    <td width="50%" valign="middle" align="center" style="border: none;">
      <h3>Confusion Matrix</h3>
      <img src="https://github.com/user-attachments/assets/90f46939-ac3a-46b5-b695-9d0749d33009" width="100%" alt="Confusion Matrix">
    </td>
  </tr>
</table>
