# Music Genre Classification – GTZAN & FMA

This project performs **music genre classification** using two popular datasets:

- **GTZAN** (10 genres, 1000 tracks)
- **FMA (Free Music Archive)** (a curated subset with 10 balanced genres)

The approach combines **traditional machine learning (ML)** methods using handcrafted audio features and **deep learning (DL)** models based on spectrograms and pretrained vision architectures (ResNet, DeiT).

---

## **1. Datasets**

### **GTZAN**

- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
- Each track is 30 seconds.
- Split into 64% train, 16% validation (if applicable), and 20% test.

### **FMA**

- Subset of FMA with 10 genres.
- Each track is trimmed to a uniform 30-second sample.
- Same processing pipeline as GTZAN.

---

## **2. Feature-Based Models (Traditional ML)**

For both datasets:

- Audio features were extracted using `librosa`:
  - Chroma (STFT, CQT, CENS)
  - MFCC (20 coefficients)
  - Spectral features (RMS, centroid, bandwidth, contrast, flatness, rolloff, tonnetz, zero-crossing rate)
  - Tempo
- Feature aggregation: mean and standard deviation over 3-second windows.
- Models tested:
  - Gaussian Naive Bayes (GNB)
  - SVM (RBF, Poly, Linear)
  - Multi-Layer Perceptron (MLP)

---

## **3. Deep Learning Models**

### **Spectrogram Processing**

- Spectrograms are precomputed (128 Mel-bands).
- **Training:** random 128-frame windows using `RandomWindowSpectrogramDataset`.
- **Evaluation:** sliding windows with majority voting using `EvaluationSpectrogramDataset`.

### **Architectures**

1. **Custom CNN**
   - 4 convolutional blocks (1 → 16 → 32 → 64 → 128 → 256 filters).
   - BatchNorm, ReLU, MaxPooling.
   - Fully connected layers with dropout.
2. **ResNet-18** (ImageNet pretrained)
   - Spectrograms resized to 224×224.
   - Last layer adapted to 10 classes.
3. **DeiT-Tiny** (Vision Transformer)
   - Similar preprocessing to ResNet.

### **Augmentation (SpecAugment)**

- Time masking (`TimeMasking`).
- Frequency masking (`FrequencyMasking`).

---

## **4. Results**

### **4.1 Machine Learning Models**

The performance of traditional ML models on both **GTZAN** and **FMA** datasets is summarized in Table 1.  
The **SVM with RBF kernel** achieved the best performance among ML baselines, with **F1 = 0.77 on GTZAN** and **F1 = 0.52 on FMA**.

**Table 1 – Evaluation of ML Models**

| **Model**   | **GTZAN Acc** | **Prec** | **Rec**  | **F1**   | **FMA Acc** | **Prec** | **Rec**  | **F1**   |
| ----------- | ------------- | -------- | -------- | -------- | ----------- | -------- | -------- | -------- |
| Gaussian NB | 0.59          | 0.60     | 0.59     | 0.59     | 0.38        | 0.38     | 0.38     | 0.34     |
| SVM (RBF)   | **0.77**      | **0.78** | **0.77** | **0.77** | **0.53**    | **0.52** | **0.53** | **0.52** |
| SVM (Poly)  | 0.71          | 0.73     | 0.71     | 0.71     | 0.49        | 0.49     | 0.49     | 0.49     |
| SVM (Lin)   | 0.72          | 0.72     | 0.71     | 0.71     | 0.48        | 0.47     | 0.48     | 0.47     |
| MLP         | 0.75          | 0.75     | 0.75     | 0.75     | 0.47        | 0.47     | 0.47     | 0.47     |

---

### **4.2 Deep Learning Models (GTZAN)**

Table 2 shows the evaluation results for **Custom CNN**, **ResNet-18**, and **DeiT-Tiny** on the GTZAN dataset.  
**ResNet-18 (no augmentation)** performs best with **F1 = 0.76** and **track accuracy of 0.84**.

**Table 2 – DL Model Evaluation on GTZAN With and Without Augmentation**

| **Model**  | **Aug.** | **Acc-Win** | **Acc-Trk** | **Prec** | **Rec**  | **F1**   |
| ---------- | -------- | ----------- | ----------- | -------- | -------- | -------- |
| Custom CNN | No       | 0.73        | 0.79        | 0.73     | 0.73     | 0.73     |
|            | Yes      | 0.71        | 0.77        | 0.73     | 0.71     | 0.71     |
| ResNet-18  | No       | **0.76**    | **0.84**    | **0.78** | **0.76** | **0.76** |
|            | Yes      | 0.75        | 0.80        | 0.77     | 0.75     | 0.75     |
| DeiT-Tiny  | No       | 0.67        | 0.74        | 0.73     | 0.67     | 0.67     |
|            | Yes      | 0.72        | 0.77        | 0.73     | 0.72     | 0.72     |

---

### **4.3 Deep Learning Models (FMA)**

Table 3 summarizes results for deep models on the **FMA dataset**.  
Again, **ResNet-18 (no augmentation)** achieved the best F1 score (**0.57**), but the overall performance is lower compared to GTZAN due to dataset complexity.

**Table 3 – DL Model Evaluation on FMA With and Without Augmentation**

| **Model**  | **Aug.** | **Acc-Win** | **Acc-Trk** | **Prec** | **Rec**  | **F1**   |
| ---------- | -------- | ----------- | ----------- | -------- | -------- | -------- |
| Custom CNN | No       | 0.49        | 0.52        | 0.48     | 0.49     | 0.48     |
|            | Yes      | 0.52        | 0.56        | 0.50     | 0.52     | 0.51     |
| ResNet-18  | No       | **0.58**    | **0.63**    | **0.58** | **0.58** | **0.57** |
|            | Yes      | 0.56        | 0.59        | 0.56     | 0.56     | 0.56     |
| DeiT-Tiny  | No       | 0.53        | 0.58        | 0.54     | 0.53     | 0.53     |
|            | Yes      | 0.51        | 0.55        | 0.50     | 0.51     | 0.49     |

---

### **4.4 Key Observations**

- **GTZAN:** ResNet-18 (no augmentation) performs best (F1 = 0.76).
- **FMA:** ResNet-18 (no augmentation) also tops with F1 = 0.57.
- **SpecAugment** improved performance slightly for **Custom CNN** but not for ResNet or DeiT.
- **SVM (RBF)** remains a strong traditional baseline for GTZAN.
