# 🧠 ResNets from Scratch (PyTorch)

This repository contains an **implementation of ResNet architectures (18, 34, 50)** built **from scratch in PyTorch**, without using `torchvision.models`.  
Each ResNet variant (ResNet-18, ResNet-34, and ResNet-50) is implemented using **modular building blocks**, such as basic and bottleneck residual blocks, to showcase a deep understanding of **Residual Learning**.

---

## 🚀 Features
- ✅ Implemented **BasicBlock** and **BottleneckBlock** manually  
- ✅ Supports **ResNet-18**, **ResNet-34**, and **ResNet-50**  
- ✅ Uses **Batch Normalization**, **Downsampling**, and **Skip Connections**  
- ✅ Fully **modular design** — easy to extend for other variants  
- ✅ Compatible with **CUDA**  
- ✅ Includes a **sanity check function** to verify architecture flow  

---

## 🧩 Architecture Overview

| Model      | Blocks per Layer | Type         | Bottleneck | Params (Approx) |
|-------------|------------------|---------------|--------------|------------------|
| ResNet-18   | [2, 2, 2, 2]     | BasicBlock    | ❌           | ~11M             |
| ResNet-34   | [3, 4, 6, 3]     | BasicBlock    | ❌           | ~21M             |
| ResNet-50   | [3, 4, 6, 3]     | Bottleneck    | ✅           | ~25M             |

---

## 🧱 Components
- **Block** → Implements 2 convolutional layers with skip connection.  
- **BottleneckBlock** → Implements 1×1 → 3×3 → 1×1 architecture used in deeper networks.  
- **Layer** → Stack of multiple blocks, handles downsampling for the first block.  
- **ResNet** → Final architecture combining all layers, pooling, and fully connected head.

---

## 🧑‍💻 Author

Jaskeerat Singh
🎯 AI/ML Engineer | Passionate about Deep Learning Architectures