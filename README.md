"the dataset is 5GB it can't be pushed otherwise it will hit the limit"
1.Dataset Preparation:
~80,000 OCT retina images
Four categories: CNV, DME, Drusen, Normal
Applied preprocessing: resizing, normalization, data augmentation
2. Model Development:
Baseline: Custom CNN → achieved moderate accuracy
Improved: ResNet-50 transfer learning → significant boost in performance
3.Training & Evaluation:
Loss function: CrossEntropy
Optimizer: Adam
Evaluation metrics: Accuracy, F1-Score, Confusion Matrix


- Tech Stack
Python, TensorFlow / Keras, OpenCV, NumPy, Matplotlib
Jupyter Notebook for experimentation
📊 Results
CNN baseline: ~70% accuracy
ResNet-50: ~90% accuracy
Visualizations: ROC curves, confusion matrix, and sample predictions
