import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

Image_width, Image_height = 299, 299
Training_Epochs = 2
Batch_size = 32
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'
num_train_samples = get_num_files(train_dir)
num_validate_samples = get_num_files(validate_dir)
num_classes = get_num_subfolders(validate_dir)
num_epochs = Training_Epochs
batch_size = Batch_size

train_img_gen = create_img_generator()
validate_img_gen = create_img_generator()

train_generatot = train_img_gen.flow_from_directory(
    train_dir,
    target_size=(Image_width, Image_height),
    batch_size=Batch_size,
    seed=42
)

validate_generator = validate_img_gen.flow_from_directory(
    validate_dir,
    target_size=(Image_width, Image_height),
    batch_size=Batch_size,
    seed=42
)

InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)
print('Inception v3 model without last FC loaded')

x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation = "relu")(x)
predictions = Dense(num_classes, activation = "softmax")(x)

print (model.summary())

model = Model(inputs = InceptionV3_base_model.input, outputs = predictions)

print('\n Fine tuning existing model')

Layers_To_Freeze = 172
for layer in model.layers[:Layers_To_Freeze]:
    layer.trainable = False
for layer in model.layers[Layers_To_Freeze:]:
    layer.trainable = True

model.compile(optimizer = SGD(lr = 0.01, momentum=0.9), loss = 'categorical_crossentropy', metrics=['accuracy'])

history_fine_tune = model.fit_generator(
    train_generatot,
    steps_per_epoch = num_train_samples // batch_size,
    epochs = num_epochs,
    validation_data = validate_generator,
    validation_steps = num_validate_samples // batch_size,
    class_weight = 'auto'
)

model.save('inception3_fine_tune_by_Kris.model')

