import tensorflow as tf
from src.model import build_model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = build_model()

# Train model
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=64)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Save trained model
model.save("saved_model/cifar10_model.h5")
print("Model saved to saved_model/cifar10_model.h5")
