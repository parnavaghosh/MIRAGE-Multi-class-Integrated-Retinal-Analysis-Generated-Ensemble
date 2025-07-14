
# 🧿 MIRAGE: Multi-class Integrated Retinal Analysis Generated Ensemble

**MIRAGE** is a deep learning-based system designed to classify retinal diseases from fundus images using transfer learning. It automates the detection of six common vision-threatening conditions with high accuracy, benchmarked against state-of-the-art medical vision models. It is optimized for low-resource settings and can be deployed as a web-based diagnostic tool.

---

## 📌 Features

- 🔍 Fine-tuned **VGG16** model using transfer learning  
- 🧠 Detection of **6 retinal diseases**:  
  `ARMD`, `Cataract`, `Diabetic Retinopathy`, `Glaucoma`, `Retinoblastoma`, `WNL`
- 📦 Custom dataset created from **RFMiD 1.0**, **RFMiD 2.0**, and web scraping
- 📊 Training, validation, and test performance tracking with visualization
- 🧪 Evaluation using confusion matrix and accuracy metrics
- 🌐 Deployable on **Hugging Face Spaces** with Gradio frontend

---

## 🧬 Diseases Classified

| Class Index | Condition                          | Description                                                                                      |
|-------------|-------------------------------------|--------------------------------------------------------------------------------------------------|
| 0           | Age-Related Macular Degeneration   | Causes central vision loss due to macula deterioration                                           |
| 1           | Cataract                           | Clouding of the eye’s lens, leads to blurry or impaired vision                                  |
| 2           | Diabetic Retinopathy               | Complication of diabetes affecting blood vessels in the retina                                   |
| 3           | Glaucoma                           | Damage to the optic nerve, often associated with high intraocular pressure                      |
| 4           | Retinoblastoma                     | Rare pediatric eye cancer, potentially fatal if untreated                                        |
| 5           | Within Normal Limit (WNL)          | Healthy retina, no abnormalities detected                                                        |

---

## 🗃️ Dataset

- The dataset was manually curated and extracted from:
  - `RFMiD 1.0`
  - `RFMiD 2.0`
  - Public domain fundus images via web scraping
- Organized into a folder structure:
  ```
  /train
  /valid
  /test
  ```

---

## ⚙️ Model Architecture

- Base model: `VGG16` pretrained on ImageNet
- Final classifier:
  - 4096 → 512 (ReLU)
  - 512 → 6 output neurons (Softmax via CrossEntropy)
- Convolutional layers are **frozen** to leverage pretrained features

---

## 🏋️ Training Pipeline

```bash
1. unzip_dataset()      → Extracts the MIRAGE dataset
2. get_data_loaders()   → Loads images using PyTorch's ImageFolder
3. setup_model()        → Loads and modifies VGG16
4. train_model()        → Training loop with early stopping & learning rate scheduler
5. plot_loss_curves()   → Loss/accuracy over epochs
6. evaluate_model()     → Final accuracy and confusion matrix
```

---

## 🖥️ Usage

### 🔧 Run Locally (in Colab or Python)
```python
python mirage_f.py
```

### 🔌 Dataset
Make sure your dataset zip is named `MIRAGE.zip` and structured with `train/`, `valid/`, `test/`.

---

## 📈 Results

| Metric        | Value         |
|---------------|---------------|
| Validation Accuracy | ~98.8%    |
| Test Accuracy       | ~98.6%    |
| F1 Score (avg)      | High across all classes |
| Med-GEMMA Benchmark | ~97.7%    |

---

## 🚀 Deployment

- Model is exported as a TorchScript `.pt` file
- Deployed via **Gradio** on **Hugging Face Spaces**
- Accepts uploaded images and returns diagnosis with links to additional resources

---

## 📂 Output Mapping

```python
{
  0: "ARMD: ... [Learn more](...)",
  1: "Cataract: ...",
  ...
}
```

This mapping is used in the Gradio frontend to return a readable diagnosis with medical references.

---

## 📎 Example Prediction

Upload a sample fundus image and get:
```
Glaucoma:

A group of eye conditions that damage the optic nerve, often due to high intraocular pressure.

[Learn more](https://www.glaucoma.org/glaucoma/what-is-glaucoma.php)
```

---

## 💡 Future Work

- Add Grad-CAM visualizations for explainability  
- Enable multi-label classification  
- Integrate with mobile fundus cameras  
- Improve robustness to unseen devices and lighting

---

## 👨‍💻 Author

Parnava Ghosh – B.Sc. (Hons) Student, Presidency University  
Developed as part of a research internship under CROW Club, with a vision to improve retinal screening in low-resource regions.
