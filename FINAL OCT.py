#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

# 1. Define paths to dataset (adjusted to your folder structure)
base_dir = "/Users/aashkagupta/Desktop/OCT"  # Replace with correct path
val_dir = os.path.join(base_dir, "val")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# 2. Verify the directories exist
print("Validation directory exists:", os.path.exists(val_dir))
print("Train directory exists:", os.path.exists(train_dir))
print("Test directory exists:", os.path.exists(test_dir))


# In[2]:


# 3. Data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)


# In[3]:


def sample_subset(directory, num_samples):
    """Sample 'num_samples' images from each class directory."""
    sampled_files = []

    # Ensure we only process directories, skipping files like .DS_Store
    class_dirs = [
        os.path.join(directory, cls) 
        for cls in os.listdir(directory) 
        if os.path.isdir(os.path.join(directory, cls))  # Only include directories
    ]

    for cls_dir in class_dirs:
        # List only files (exclude hidden/system files)
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)) and not f.startswith('.')]
        
        # Handle cases where there are fewer files than needed samples
        if len(files) < num_samples:
            print(f"Warning: {cls_dir} contains fewer files than {num_samples}. Using all available files.")
            sampled_files.extend([os.path.join(cls_dir, f) for f in files])
        else:
            sampled_files.extend([os.path.join(cls_dir, f) for f in random.sample(files, num_samples)])

    return sampled_files

# Define your directory paths
base_dir = "/Users/aashkagupta/Desktop/OCT"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "val")

# Check if directories exist
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")
print(f"Validation directory exists: {os.path.exists(val_dir)}")

# Sample the datasets (8000 from train, 800 from test, and all from val)
train_samples = sample_subset(train_dir, 8000 // len(os.listdir(train_dir)))
test_samples = sample_subset(test_dir, 800 // len(os.listdir(test_dir)))

# For validation, take all images
val_samples = []
val_class_dirs = [
    os.path.join(val_dir, cls) 
    for cls in os.listdir(val_dir) 
    if os.path.isdir(os.path.join(val_dir, cls))
]
for cls_dir in val_class_dirs:
    val_samples.extend([
        os.path.join(cls_dir, f) 
        for f in os.listdir(cls_dir) 
        if os.path.isfile(os.path.join(cls_dir, f)) and not f.startswith('.')
    ])

print(f"Number of train samples: {len(train_samples)}")
print(f"Number of test samples: {len(test_samples)}")
print(f"Number of validation samples: {len(val_samples)}")


# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create an ImageDataGenerator for data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values
    rotation_range=20,           # Randomly rotate images
    width_shift_range=0.2,       # Randomly shift images horizontally
    height_shift_range=0.2,      # Randomly shift images vertically
    shear_range=0.2,             # Shear transformation
    zoom_range=0.2,              # Randomly zoom
    horizontal_flip=True,        # Randomly flip images
    fill_mode='nearest'          # Fill empty pixels
)

# Load and preprocess training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Change to 'binary' if you have 2 classes
    shuffle=True
)

# Load and preprocess validation data (without augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load and preprocess test data (without augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Create a CNN model using a pre-trained ResNet50 as base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_data.class_indices), activation='softmax'))  # Change to 'sigmoid' for binary

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=2,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    validation_steps=val_data.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr]
)


# In[5]:


# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[7]:


# After training, print the final training and validation accuracies
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Evaluate the model on test data to print the test accuracy
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


# In[ ]:




