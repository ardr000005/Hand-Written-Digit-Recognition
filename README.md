```markdown
# DigitVision: MNIST Handwritten Digit Recognition using TensorFlow

This project demonstrates how to train a deep learning model using **TensorFlow** and **Keras** to recognize handwritten digits from the **MNIST dataset**.  
It walks through data loading, preprocessing, model building, training, evaluation, and visualization — making it a great introduction to neural networks and computer vision.

---

## 🧠 Features

- Uses the **MNIST dataset** of 70,000 grayscale images (28×28 pixels).
- Builds a **Sequential Neural Network (MLP)** using TensorFlow/Keras.
- Includes **Dropout regularization** to reduce overfitting.
- Provides **training history**, **test accuracy**, and **confusion matrix** visualization.
- Saves the trained model as `mnist_model.h5`.

---

## 🗂️ Project Structure

```
DigitVision/
├── Handwritten Digit.ipynb          # Main Jupyter notebook
├── mnist_model.h5     # Saved trained model (after running)
├── README.md          # Project documentation
└── requirements.txt   # Dependencies
```

---

## ⚙️ Requirements

Install the required Python packages:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Example `requirements.txt`:

```
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ardr000005/Hand-Written-Digit-Recognition.git
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook Handwritten Digit.ipynb
   ```

3. **Run all cells** to:
   - Load and normalize the MNIST dataset
   - Train the model
   - Evaluate accuracy and visualize predictions

---

## 🧩 Model Architecture

| Layer Type | Details |
|------------|---------|
| Input | 28×28 Flattened |
| Dense | 128 neurons, ReLU activation |
| Dropout | 0.2 (for regularization) |
| Output | 10 neurons, Softmax activation |

**Optimizer**: Adam  
**Loss**: Sparse Categorical Crossentropy  
**Metrics**: Accuracy

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~97% |
| Test Accuracy | ~97–98% |

**Includes**:
- Confusion matrix heatmap using seaborn
- Sample predictions visualization for first few test images

---

## 📈 Example Output

- Model summary printed in notebook
- Training vs validation accuracy plot (via model.fit)
- Confusion matrix for test set
- Predictions displayed using matplotlib

---

## 🧪 Future Improvements

- Convert to CNN for higher accuracy
- Deploy model using Streamlit or Flask
- Integrate TensorFlow Lite for on-device inference

---

## 🧑‍💻 Author

**Aravind**  

---

## 🪪 License

This project is licensed under the **MIT License**.
```
