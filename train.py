import os
import argparse
import tensorflow as tf
from models.model import build_adf_net
from utils.preprocessing import get_data_generators
from utils.visualization import plot_training_history

def main():
    parser = argparse.ArgumentParser(description="Train ADF-Net on COVID-19 CT Scan Dataset")
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for Adam optimizer')
    parser.add_argument('--model_save_path', type=str, default='best_model.h5', help='Filepath to save the best model')
    parser.add_argument('--plots_dir', type=str, default='results', help='Directory to save training plots')
    args = parser.parse_args()

    # Define paths
    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'validation')
    # Fallback to 'val' if 'validation' does not exist
    if not os.path.exists(val_dir) and os.path.exists(os.path.join(args.dataset_dir, 'val')):
        val_dir = os.path.join(args.dataset_dir, 'val')
    test_dir = os.path.join(args.dataset_dir, 'test')

    print(f"Loading data from:\n - Train: {train_dir}\n - Val: {val_dir}\n - Test: {test_dir}")

    # Load Data Generators
    train_gen, val_gen, test_gen = get_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        target_size=(224, 224),
        dataset_dir=args.dataset_dir
    )

    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())
    print(f"Detected Classes: {class_names} (Total: {num_classes})")

    # Build ADF-Net Model
    print("Building ADF-Net model architecture...")
    model = build_adf_net(input_shape=(224, 224, 3), num_classes=num_classes)
    
    # Print Model Summary
    model.summary()

    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Configure Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]

    # Train Model
    print("Starting ADF-Net training process...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Plot and Save Curves
    print(f"Training completed. Plotting accuracy and loss curves to: {args.plots_dir}")
    plot_training_history(history, args.plots_dir)
    print("Training process finished successfully. Best model saved to disk.")

if __name__ == '__main__':
    main()
