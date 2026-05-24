# Retail Product stacking system using COMPUTER VISION

AI-powered retail shelf product detection and grouping system using YOLOv8, CLIP, and DBSCAN.

---

## Features

- Product Detection using YOLOv8
- Product Embedding using CLIP
- Similar Product Grouping using DBSCAN
- Streamlit-based Web Interface
- Custom-trained Retail Detection Model

---

## Tech Stack

- Python
- YOLOv8
- OpenCV
- CLIP
- DBSCAN
- Streamlit

---

## Project Pipeline

Shelf Image  
→ Product Detection  
→ Product Cropping  
→ Embedding Generation  
→ Product Clustering  
→ Visualization

---

## Dataset

SKU-110K Retail Dataset

---

## Deployment Approach

Initially, FastAPI was considered for building REST API endpoints for the project.  
However, for easier visualization, testing, and interactive demonstration, the project was finally deployed using Streamlit-based UI integration.

This allows users to directly upload retail shelf images and visualize grouped product outputs in real time.

---

## AI Usage

AI tools were used during development for debugging assistance, implementation guidance, and workflow optimization.  
However, the complete project pipeline, training process, integration, testing, debugging, and final customization were performed manually.

---

## Run Project

### Install Requirements

pip install -r requirements.txt

### Run App

streamlit run streamlit_app.py

---

## Project Structure

detector/  
embeddings/  
grouping/  
visualization/  
outputs/  
streamlit_app.py  
train.py  

---

## Author

Vedant Verma
