import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir, img_size=(150, 150), batch_size=32):
    """
    Create training and validation data generators with data augmentation.

    Parameters:
    - data_dir (str): Path to the data directory.
    - img_size (tuple): Target size of the images.
    - batch_size (int): Number of images to be yielded from the generator per batch.

    Returns:
    - train_generator: Data generator for training data.
    - val_generator: Data generator for validation data.
    """
    # Ensure the data directory exists
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", data_dir))
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise FileNotFoundError(f"Data directory is empty or does not exist: {data_dir}")
    if not os.listdir(data_dir):
        raise FileNotFoundError(f"No subdirectories found in the data directory: {data_dir}")
    
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.listdir(subdir_path):
            raise FileNotFoundError(f"No images found in the subdirectory: {subdir_path}")
    
    # Create an ImageDataGenerator with data augmentation and validation split
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Add brightness augmentation
        zoom_range=0.1  # Add zoom augmentation
    )
    
    # Create a training data generator
    train_generator = datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Create a validation data generator
    val_generator = datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Log data distribution
    print("Training class distribution:", train_generator.class_indices)
    print("Validation class distribution:", val_generator.class_indices)
    
    return train_generator, val_generator
