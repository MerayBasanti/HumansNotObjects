# HumansNotObjects
A deep learning image classifier that distinguishes human faces from non-human objects using MobileNetV2, with adversarial robustness testing (FGSM).

# Human vs Object Image Classifier 🚀

A deep learning image classifier that distinguishes **human faces** (from the LFW dataset) from **non-human objects** (from an additional dataset).  
It uses **Transfer Learning** with **MobileNetV2** and includes **adversarial robustness testing** with the **Fast Gradient Sign Method (FGSM)**.

---

## 📌 **Project Highlights**

✅ Uses **Labeled Faces in the Wild (LFW)** for human face images.  
✅ Uses an additional objects dataset for non-human images.  
✅ Fine-tunes **MobileNetV2** for binary classification: *Is it human or not?*  
✅ Includes clear **train**, **validation**, and **test** splits.  
✅ Tests the model’s robustness with **FGSM adversarial attacks**.  
✅ Provides defense suggestions: adversarial training, preprocessing, distillation.

---

## 🗂️ **Project Structure**

```plaintext
📂 lfw-deepfunneled/
   └── [folders of person images]

📂 objects/
   └── [folders or images of objects]

📄 notebook.ipynb      # Main Colab / Jupyter Notebook
📄 my_mobilenetv2_classifier.keras  # Saved trained model
📄 README.md           # This file
```

---

## ⚙️ **How to Run**

### ✅ 1️⃣ Prepare Datasets

* Place the **LFW** dataset under `lfw-deepfunneled/` (organized in folders by person).
* Place your **object images** in a separate folder.

### ✅ 2️⃣ Train the Model

* Use the notebook to:

  * Load data with `pandas` + `ImageDataGenerator`.
  * Create `train`, `validation`, and `test` splits.
  * Fine-tune **MobileNetV2** on your binary labels.
  * Save your trained model:

    ```python
    model.save('my_mobilenetv2_classifier.keras')
    ```

### ✅ 3️⃣ Test Adversarial Robustness

* Load the saved model:

  ```python
  from tensorflow.keras.models import load_model
  model = load_model('my_mobilenetv2_classifier.keras')
  ```

* Run the **FGSM loop** to generate adversarial examples and measure the impact.

---

## ✅ **Final Adversarial Results**

After testing with **FGSM attack** (`epsilon = 0.01`):

* **Total test samples:** `6485`
* **Clean accuracy:** `100%`
* **Adversarial accuracy:** `91.13%`

**Adversarial confusion matrix:**

|                   | Predicted Object | Predicted Human |
| ----------------- | ---------------- | --------------- |
| **Actual Object** | 4436             | 64              |
| **Actual Human**  | 511              | 1474            |

**Interpretation:**

* The model achieves perfect accuracy on clean test images.
* Under FGSM, the accuracy drops to ~91%, with human faces more affected than objects.
* About **511 human faces** were misclassified as objects under adversarial noise — a common vulnerability for gradient-based attacks.

---

## 🔐 **Adversarial Robustness Tips**

✅ **Adversarial training**: Fine-tune your model on a mix of clean + adversarial examples for better resilience.

✅ **Input preprocessing defenses**: Add denoising, compression, or random noise layers to disrupt gradient-based attacks.

✅ **Stronger attack testing**: Evaluate with **PGD**, **BIM**, or **C&W** for deeper robustness insights.

---

## 🧩 **Dependencies**

* `TensorFlow >= 2.x`
* `scikit-learn`
* `pandas`
* `numpy`

Install with:

```bash
pip install tensorflow scikit-learn pandas numpy
```

---

## 🚀 **Next Steps**

* Add **adversarial training** to improve robustness.
* Experiment with stronger attacks and defense strategies.
* Deploy the model with **TensorFlow Lite**, **ONNX**, or an edge device.

---

## ✨ **Credits**

* **LFW Dataset** — University of Massachusetts Amherst.
* **MobileNetV2** — Pretrained on ImageNet.

---

## 📬 **Contact**

Questions or suggestions?
Feel free to open an issue or pull request!
Happy experimenting — keep your models robust! 🎉
