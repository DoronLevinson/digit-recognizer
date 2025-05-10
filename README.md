# Digit Recognizer Web App

An interactive Streamlit web app that predicts hand-drawn digits (0–9) using multiple machine learning models:

- 🟦 **Convolutional Neural Network (CNN)**
- 🟪 **Fine-Tuned Vision Transformer (CLIP)**
- 🟥 **Deep Neural Network (DNN)**
- 🟩 **K-Nearest Neighbors (KNN)**

Each model provides a real-time prediction along with its confidence level.

---

## 📁 Project Structure

```
.
├── app.py                         # Main Streamlit app
├── inference.py                  # Shared inference and utility functions
├── cnn_model.py                  # CNN model loader and predictor
├── dnn_model.py                  # DNN model loader and predictor
├── vit_classifier_model.py       # CLIP ViT model loader and predictor
├── models/
│   ├── cnn_model.pth             # Saved CNN model
│   ├── dnn_model.pth             # Saved DNN model
│   ├── clip_digit_classifier.pth # Saved ViT model
│   ├── X_knn.csv                 # KNN feature vectors
│   ├── y_knn.csv                 # KNN labels
├── requirements.txt              # Python dependencies
├── Dockerfile                    # (Optional) Docker configuration
├── .gitignore                    # Files to ignore in Git
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DoronLevinson/digit-recognizer.git
cd digit-recognizer
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you get an error about missing `sklearn`, also run:

```bash
pip install scikit-learn
```

---

## Run the App Locally

```bash
streamlit run app.py
```

Open your browser to [http://localhost:8502](http://localhost:8502)

---

## How to Use

1. Draw a digit (0–9) in the canvas area.
2. Toggle which models to display in the sidebar.
3. View the predictions and confidence bars on the right side.
4. Optionally, submit feedback on incorrect predictions.

---

##Author

[Doron Levinson](https://www.linkedin.com/in/doron-levinson/) 
[GitHub Profile](https://github.com/DoronLevinson)

---