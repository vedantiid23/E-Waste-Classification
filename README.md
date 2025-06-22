E-Waste Generation Classification
This project implements an image classification model to identify types of e-waste using deep learning. Leveraging transfer learning with EfficientNetV2B0, the model is trained on a custom dataset consisting of three primary classes: battery, cable, and mobile.

📁 Project Structure
css
Copy
Edit
AICTE-Internship-main/
└── P3 - E-Waste Generation Classification/
    ├── modified-dataset/
    │   ├── train/
    │   │   ├── battery/
    │   │   ├── cable/
    │   │   └── mobile/
    │   ├── val/
    │   │   ├── battery/
    │   │   ├── cable/
    │   │   └── mobile/
    │   └── test/
    │       ├── battery/
    │       ├── cable/
    │       └── mobile/
    ├── main.py
    └── README.md
🚀 Features
✅ Transfer learning with EfficientNetV2B0

✅ Image classification across 3 e-waste categories

✅ Model training with real-time performance visualization

✅ Evaluation using accuracy, precision, recall, and F1-score

✅ Confusion matrix generation and report summary

✅ Early stopping and learning rate scheduling

✅ Model persistence with .h5 saving

🧠 Model Workflow
Data Preparation
Dataset split into train, val, and test using ImageDataGenerator.

Model Architecture
Pretrained EfficientNetV2B0 base with custom classification head.

Training
Model is trained with callbacks like EarlyStopping and ReduceLROnPlateau.

Evaluation
Metrics include confusion matrix and classification report (scikit-learn).

Model Saving
Trained model saved as e_waste_classifier.h5.

📦 Dependencies
Install the required libraries:

bash
Copy
Edit
pip install tensorflow matplotlib seaborn scikit-learn
🛠 How to Run
Run the training and evaluation script:

bash
Copy
Edit
python main.py
📊 Output
Terminal output with training metrics and final evaluation

Confusion matrix plot for test set

Saved model: e_waste_classifier.h5

🖼 Class Labels
Class	Description
battery	Batteries, cells, power packs
cable	USB cables, wires, connectors
mobile	Smartphones, feature phones

📈 Optional Enhancements
You can further improve this project by:

Adding TensorBoard support for detailed training logs

Exporting metrics to CSV or JSON

Including Grad-CAM visualizations for model interpretability

Packaging the model for web/mobile inference

👥 Contributors
AICTE Internship Team – P3 Project Group

📄 License
This project is for academic and research purposes under the AICTE internship initiative. Please contact the team for reuse or distribution permissions.
