---

# 🧬 Microscopic Object Detection System

## 📌 Overview

This project is a **Streamlit-based web application** that performs **microscopic cell component detection** using a **YOLO ONNX model (Ultralytics)**.
The system identifies biological structures such as **nucleus, vacuole, ribosome**, or any classes you trained.

After detection, the processed image and corresponding object counts are **stored securely in MongoDB Atlas** using GridFS. This enables persistent storage of detection results for analysis, research, and audit purposes.

The app provides an intuitive interface for uploading microscope images, running real-time detection, and storing results seamlessly.

---

## ✨ Key Features

### 🔬 Microscopic Object Detection

* Powered by **Ultralytics YOLO ONNX**
* Detects and counts target cellular components
* Draws bounding boxes and labels on the image

### ⚙️ ONNX Runtime + Ultralytics Wrapper

* Fast inference
* Lightweight cross-platform model
* CPU-friendly deployment using ONNX

### 🧵 Streamlit Web Interface

* Upload an image or capture via camera
* Run inference with adjustable confidence threshold
* View annotated detection results instantly
* Clean, user-friendly UI

### ☁️ MongoDB Atlas Integration

* Stores detection results in the cloud
* Uses **GridFS** for saving processed images
* Saves:

  * image
  * class counts
  * timestamp
  * model info/metadata
* Fully secured using Streamlit Secrets & Atlas Authentication

---

## 🛠 Technology Stack

### **Frontend / App Layer**

* **Streamlit** – lightweight, fast UI for ML apps
* **Pillow (PIL)** – image drawing and processing

### **Model**

* **Ultralytics YOLO (ONNX)**
* Exported using:

  ```python
  model.export(format="onnx")
  ```

### **Backend / Storage**

* **MongoDB Atlas** – cloud-based NoSQL database
* **GridFS** – stores high-resolution annotated images
* **PyMongo** – Python driver for MongoDB

### **Supporting Tools**

* NumPy
* Requests
* Python 3.10+

---

## 🚀 Use Cases

### 🧫 **Lab Image Analysis**

Automatically identify and quantify components inside microscopic images.

### 🏥 **Research & Healthcare**

Useful for:

* Cell counting
* Pathology
* Morphological studies
* Automated microscopy workflows

### 📚 **Education**

Helps students understand cell structures with automated visual assistance.

### 🧪 **Dataset Creation**

Stores processed images and counts for future ML or statistical research.

---

## ⭐ Benefits

### ⏱ Faster Analysis

Automates manual counting and annotation of microscope images.

### 📦 Cloud Storage

Detection results stored in **MongoDB Atlas** for long-term access.

### 💻 No Local Setup Needed

Runs fully in a web browser thanks to Streamlit.

### ⚡ Lightweight Deployment

ONNX model ensures portability across CPU-only systems.

### 🔐 Secure

Secrets handled via Streamlit Secrets Manager.

---

## 🌐 API / External Services

### 🔹 **MongoDB Atlas**

Used for securely storing detection results and images using GridFS.
Provides global availability, scaling, and automatic backups.
👉 [https://www.mongodb.com/atlas/database](https://www.mongodb.com/atlas/database)

### 🔹 **Ultralytics YOLO**

Model training and ONNX export.
Simplifies object detection pipelines.
👉 [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

---

## 📂 Project Structure (Typical)

```
├── app.py
├── best.onnx
├── requirements.txt
├── README.md
└── secrets.toml
```

---

## 🔧 Environment Setup

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app.py
```

---

## 🔐 Streamlit Secrets Example

`.streamlit/secrets.toml`

```toml
[mongo]
uri = "YOUR_MONGODB_ATLAS_URI"
```

---

## 💾 Database Entry Example

Each detection stores:

```json
{
  "timestamp": "2025-01-01T10:00:00Z",
  "counts": { "nucleus": 10, "vacuole": 5 },
  "model": "best.onnx",
  "img_gridfs_id": "65b7f35d912da...."
}
```

Image is saved via **GridFS** for high-resolution storage.

---

## 📸 Demo Flow

1. Upload microscope image
2. App runs YOLO ONNX inference
3. Bounding boxes drawn
4. Result image saved to MongoDB
5. Document (image_id + counts) also stored
6. Confirmation message shown to user

---

## 🧑‍🔬 Conclusion

This project provides a complete pipeline for **microscopic object detection**, **visualization**, and **cloud storage**.
It combines the power of **YOLO ONNX**, **Streamlit**, and **MongoDB Atlas** into a scalable, easy-to-deploy system for labs, researchers, and educational institutions.

