"""
Enhanced training configuration for higher accuracy (95%+)
This script modifies training parameters for better convergence
"""

# Key improvements for 95%+ accuracy:

TRAINING_IMPROVEMENTS = {
    # 1. More aggressive data augmentation
    "augmentation": {
        "rotation_range": 7,        # Increase from 5
        "perspective_prob": 0.4,    # Increase from 0.2  
        "noise_prob": 0.3,         # Increase from 0.2
        "elastic_transform": True,  # Add elastic deformation
    },
    
    # 2. Better optimization
    "training": {
        "learning_rate": 5e-4,      # Lower LR for fine details
        "weight_decay": 5e-4,       # Stronger regularization
        "batch_size": 16,           # Smaller batches for stability
        "epochs": 200,              # Train longer
        "patience": 20,             # More patience
    },
    
    # 3. Enhanced model
    "model": {
        "dropout": 0.3,             # Increase dropout
        "lstm_layers": 3,           # Add one more LSTM layer
        "attention_heads": 2,       # Multi-head attention
        "feature_channels": [64, 128, 256, 512, 1024]  # Deeper network
    },
    
    # 4. Better preprocessing
    "preprocessing": {
        "target_size": (128, 512),  # Higher resolution
        "contrast_enhancement": True,
        "noise_reduction": True,
        "edge_enhancement": True,
    }
}

print("Enhanced training configuration for 95%+ accuracy")
print("Apply these settings for next training round")