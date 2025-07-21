
# 🐶 Dog Breed Classification with ResNet-18 and ResNet-50

This project compares two pre-trained models — **ResNet-18** and **ResNet-50** from [Hugging Face Transformers](https://huggingface.co/models) — to classify dog breeds using the [Stanford Dogs dataset](https://www.tensorflow.org/datasets/catalog/stanford_dogs). The models were evaluated in terms of prediction accuracy.

---

## 🔍 Project Overview

- 📦 **Dataset**: `stanford_dogs` from TensorFlow Datasets
- 🧠 **Models**: `microsoft/resnet-18` and `microsoft/resnet-50`
- 🧰 **Tools**: PyTorch, Hugging Face Transformers, TensorFlow Datasets
- 🎯 **Goal**: Evaluate the performance of ResNet-18 vs ResNet-50 on a dog breed classification task.

---

## 🧪 Sample Inference

For each image:
```python
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = image_processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

---

## 📈 Accuracy Comparison

The predictions were compared to the true labels extracted from the image filenames. Accuracy was computed with fuzzy matching:

| Model      | Accuracy |
|------------|----------|
| ResNet-18  | ✅ (insert actual result) |
| ResNet-50  | ✅ (insert actual result) |

> Accuracy was calculated by checking full or partial string matches between predicted labels and true labels.

---

## 📦 Installation

```bash
pip install torch torchvision
pip install transformers tensorflow_datasets
```

---

## 📂 Project Structure

```
resnet-dog-classification/
├── resnet_comparison.ipynb
├── README.md
```


