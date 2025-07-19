# MIRAGE: A Transfer Learning-Based Framework for Retinal Disease Detection and Benchmarking Against Med-GEMMA in Low-Resource Settings

## Authors

- **Parnava Ghosh**, Presidency University, Kolkata  
- **Dayita Singha**, Heritage Institute of Technology, Kolkata  
- **Ananya Mondal**, Presidency University, Kolkata  
- **Shaheli Manna**, Presidency University, Kolkata  
- **Anusha Mondal**, Presidency University, Kolkata  
- **Madhhyala Gayathri**, Vignan's Institute of Management and Technology for Women  

**Project Guide**: Rithesh Sreenivasan  
**Internship Period**: 19th May 2025 – 15th July 2025  
**Submitted To**: IDEAS – ISI Kolkata  

---

## 🌟 Abstract

This project presents **MIRAGE**, a deep learning pipeline for detecting six retinal diseases from fundus images. We curated a custom dataset by merging RFMiD 1.0, RFMiD 2.0, and web-scraped data. A fine-tuned VGG16 model achieved **98.8% accuracy**, outperforming Google's Med-GEMMA model (**97.7%**). The model was deployed via **Hugging Face Spaces** with a responsive web frontend, providing real-time diagnosis in low-resource settings.

---

## 🎯 Objectives

- Build an AI model for detecting six retinal conditions: **WNL**, **Cataract**, **Glaucoma**, **Retinoblastoma**, **Diabetic Retinopathy (DR)**, **ARMD**
- Create the **MIRAGE dataset** from RFMiD and public sources
- Fine-tune **VGG16** and benchmark against **Med-GEMMA**
- Enable real-time web-based diagnosis for LMICs and rural India
- Validate deployment feasibility in frontline healthcare systems

---

## 📚 Literature Review Highlights

- **Semi-supervised GANs** and **Contrastive Learning** boost performance with minimal labels (Lecouat 2018, Kukačka 2021)
- **Transfer Learning** with VGG16, InceptionV3, DenseNet yields >98% accuracy in medical imaging (Mutawa et al. 2023)
- Task-specific fine-tuning outperforms zero-shot models like Med-GEMMA in low-resource datasets

---

## 🛠️ Methodology Overview

### Dataset: MIRAGE

- Sourced from: RFMiD 1.0, RFMiD 2.0, web-scraped data
- Classes:
  - **WNL** (Healthy)
  - **Cataract**
  - **Glaucoma (GL)**
  - **Retinoblastoma (RB)**
  - **Diabetic Retinopathy (DR)**
  - **Age-related Macular Degeneration (ARMD)**

### Data Preprocessing

- **Resize**: 224x224  
- **Normalization**: ImageNet mean & std  
- **Augmentation**: Flip, Rotate, Brightness, Contrast  
- **Split**: Train 70%, Validation 15%, Test 15%  
- **Class Balance**: Oversampling & Augmentations

---

## 🧠 Model Training (VGG16)

- Architecture: VGG16 + Custom Head (Dropout + FC)
- Strategy:
  - Phase 1: Freeze conv layers, train FC
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` (lr=1e-4, weight decay=1e-4)
- Epochs: 30 | Batch Size: 16
- Validation Checkpoints & Confusion Matrix Analysis

---

## 🧪 Benchmark: Med-GEMMA vs VGG16

| Metric            | VGG16 (Fine-Tuned) | Med-GEMMA (Zero-Shot) |
|-------------------|--------------------|------------------------|
| Accuracy          | 98.8%              | 97.0%                  |
| Macro F1-Score    | 0.99               | 0.97                   |
| Best Class        | Retinoblastoma     | Retinoblastoma         |
| Training Needed   | ✅ Yes             | ❌ No                  |
| Deploy Locally    | ✅ Yes             | ❌ Cloud Only          |

**Inference with Prompts** (for Med-GEMMA):  
> “You are a highly capable medical vision-language AI assistant specialized in ophthalmology…”

---

## 🌐 Deployment

- Platform: Hugging Face Spaces  
- Frontend: HTML + CSS + JS  
- Backend: PyTorch API  
- Features:
  - Upload fundus image
  - Get predicted disease label
  - Clean UI, mobile compatible

---

## 📊 Results

### VGG16 (on Test Set)

- **Overall Accuracy**: 98.8%
- **Retinoblastoma**: 100% precision & recall
- Minor confusion: ARMD vs WNL

### Metrics (Summary)

| Class       | Precision | Recall | F1 Score | Support |
|-------------|-----------|--------|----------|---------|
| ARMD        | 0.98      | 0.96   | 0.97     | 248     |
| Cataract    | 1.00      | 0.99   | 0.99     | 228     |
| DR          | 0.99      | 1.00   | 0.99     | 233     |
| GL          | 0.99      | 0.99   | 0.99     | 237     |
| RB          | 1.00      | 1.00   | 1.00     | 221     |
| WNL         | 0.98      | 0.99   | 0.98     | 250     |

---

## ✅ Conclusion

- Fine-tuned VGG16 significantly outperforms Med-GEMMA for Indian retinal data
- High performance even on rare conditions like Retinoblastoma
- Deployable, interpretable, and cost-effective for rural diagnostics
- A step toward AI-powered universal eye care

---

## 🔭 Future Scope

- Add **Grad-CAM** explainability  
- Enable **Multi-label detection**  
- Expand to **mobile app** (Android/iOS)  
- Embed in **fundus camera hardware**  
- Support **EHR integration**, **real-time feedback loop**  

---

## 🔗 Links

- 🔍 **Web App**: [miragemodel-mirage.hf.space](https://miragemodel-mirage.hf.space/)  
- 💻 **GitHub**: [MIRAGE GitHub Repo](https://github.com/parnavaghosh/MIRAGE-Multi-class-Integrated-Retinal-Analysis-Generated-Ensemble)  
- 📂 **RFMiD 1.0**: [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/retinal-fundus-images)  
- 📂 **RFMiD 2.0**: [Roboflow](https://universe.roboflow.com/mad-18k7g/rfmid-20-extended)  
- 🧠 **Med-GEMMA**: [HuggingFace Inference](https://huggingface.co/google/med-gemma)

---

## 📄 Cited References

- Lecouat et al. (2018). *Semi-supervised GAN for DR detection* – [arXiv](https://arxiv.org/abs/1812.07832)
- Kukačka et al. (2021). *Contrastive learning for segmentation* – [arXiv](https://arxiv.org/abs/2208.02408)
- Alam et al. (2022). *Contrastive pretraining for DR* – [PubMed](https://pubmed.ncbi.nlm.nih.gov/36571941/)
- Mutawa et al. (2023). *VGG16, InceptionV3 on DR* – [MDPI](https://www.mdpi.com/2076-3417/13/9/5685)
- Hammoudi et al. (2024). *DR detection in remote areas* – [Springer](https://link.springer.com/article/10.1007/s44196-024-00520-w)

---

## 📁 Appendix Summary

- **Model**: VGG16 (ImageNet pre-trained)  
- **Loss**: CrossEntropyLoss  
- **Augmentation**: Flip, Rotate, Brightness  
- **Splitting**: Stratified 70:15:15  
- **Tools**: PyTorch, Hugging Face, Gradio, Roboflow