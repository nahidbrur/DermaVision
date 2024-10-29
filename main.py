import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def train(args):
    IMG_SIZE = 224
    NUM_CLASSES = 7


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        args.test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.BATCH_SIZE,
        class_mode='categorical'
    )

    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)


    model = Model(inputs=base_model.input, outputs=predictions)


    model.compile(optimizer=args.optimizer(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = "./checkpoint/densenet_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    # history = model.fit(
    #     train_generator,
    #     epochs=args.EPOCHS,
    #     validation_data=val_generator,
    #     steps_per_epoch=train_generator.samples // args.BATCH_SIZE,
    #     validation_steps=val_generator.samples // args.BATCH_SIZE,
    #     callbacks=[checkpoint]
    # )

    model.load_weights('./checkpoint/densenet_model.h5')
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    # True labels
    y_true = val_generator.classes

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)

    # Classification report
    target_names = list(val_generator.class_indices.keys())  # Get class names from generator
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Skin disease classification")
    parser.add_argument("--train_dir", type=str, default="./dataset/train", help="Path of the train directory")
    parser.add_argument("--test_dir", type=str, default="./dataset/test", help="Path of the test directory")
    parser.add_argument("--optimizer", type=str, default=Adam, help="Optimizer")
    parser.add_argument("--BATCH_SIZE", type=str, default=32, help="Batch_size")
    parser.add_argument("--EPOCHS", type=str, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=str, default=0.001, help="Learning rate")

    args = parser.parse_args()

    train(args)