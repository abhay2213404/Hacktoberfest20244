import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess data (normalize to [0, 1] range)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding of labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Use data augmentation to generate more training data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# Load the pre-trained ResNet50 model with ImageNet weights, excluding the top layer
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the ResNet50 layers to prevent them from being trained
for layer in resnet.layers:
    layer.trainable = False

# Add custom classification layers on top of ResNet50
x = Flatten()(resnet.output)
x = Dense(512, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=resnet.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=10,
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the trained model
model.save('resnet_cifar10_model.h5')
