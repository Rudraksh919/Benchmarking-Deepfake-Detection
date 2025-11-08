# -*- coding:utf-8 -*-
"""
Deepfake Detection Classifiers

This module contains implementations of deep learning models for deepfake detection:
- Meso4: Lightweight CNN designed for mesoscopic deepfake detection
- XceptionClassifier: Modified Xception architecture for binary classification

Author: EE656 Course Project
Date: 2025
"""

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception

# Image dimensions for input
IMGWIDTH = 256


class Classifier:
    """
    Base classifier class providing common interface for all models.
    
    Attributes:
        model: The underlying Keras model
    """
    
    def __init__():
        self.model = 0
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x: Input data (numpy array or tensor)
            
        Returns:
            Predictions as numpy array
        """
        if hasattr(x, '__len__') and len(x) == 0:
            return []
        return self.model.predict(x)

    
    def fit(self, x, y):
        """
        Train the model on a single batch.
        
        Args:
            x: Input training data
            y: Target labels
            
        Returns:
            Training loss and metrics
        """
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        """
        Evaluate model accuracy on test data.
        
        Args:
            x: Input test data
            y: True labels
            
        Returns:
            Test loss and accuracy
        """
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        """
        Load model weights from file.
        
        Args:
            path: Path to the weights file (.h5 format)
        """
        self.model.load_weights(path)




class Meso4(Classifier):
    """
    Meso4 architecture for deepfake detection.
    
    A lightweight CNN with 4 convolutional layers designed to detect
    mesoscopic properties of deepfakes. The architecture uses batch
    normalization and dropout for regularization.
    
    Args:
        learning_rate (float): Learning rate for Adam optimizer. Default: 0.001
        
    Reference:
        "Towards Benchmarking and Evaluating Deepfake Detection" (IEEE)
    """
    
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def init_model(self):
        """
        Initialize the Meso4 model architecture.
        
        Returns:
            Compiled Keras model
        """ 
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(negative_slope=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)


class XceptionClassifier(Classifier):
    """
    Modified Xception architecture for deepfake detection.
    
    Uses the Xception base architecture with depthwise separable convolutions
    and adds a custom classification head for binary deepfake detection.
    
    Args:
        learning_rate (float): Learning rate for Adam optimizer. Default: 0.001
        
    Reference:
        "Towards Benchmarking and Evaluating Deepfake Detection" (IEEE)
    """
    
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def init_model(self):
        """
        Initialize the Xception model with custom classification head.
        
        Returns:
            Compiled Keras model with Xception base and sigmoid output
        """
        base_model = Xception(weights=None, include_top=False, input_shape=(IMGWIDTH, IMGWIDTH, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=base_model.input, outputs=x)