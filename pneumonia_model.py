import os
import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tf_explain.core.grad_cam import GradCAM


kaggle_path = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", ".kaggle", "kaggle.json")
if not os.path.exists(kaggle_path):
    raise FileNotFoundError(f"❌ kaggle.json not found at {kaggle_path}")

with open(kaggle_path, "r") as f:
    creds = json.load(f)

os.environ["KAGGLE_USERNAME"] = creds["username"]
os.environ["KAGGLE_KEY"] = creds["key"]


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

dataset = "paultimothymooney/chest-xray-pneumonia"
zip_path = "chest-xray-pneumonia.zip"
extract_path = "chest_xray"

if not os.path.exists(zip_path):
    print("📦 Downloading dataset from Kaggle...")
    api.dataset_download_files(dataset, path=".", unzip=False)

if not os.path.exists(extract_path):
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


base_dir = os.path.join(extract_path, "chest_xray")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")


train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
).flow_from_directory(
    train_dir, target_size=(224, 224), class_mode='binary'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(224, 224), class_mode='binary'
)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_gen, validation_data=val_gen, epochs=5)


model.save("pneumonia_model.keras", save_format='keras')
model.save("pneumonia_model.h5", save_format='h5')
print("\n✅ Model saved as .keras and .h5")


output_path = os.getcwd()
print(f"\n📂 Opening output folder: {output_path}")
subprocess.Popen(f'explorer "{output_path}"')


img_path = os.path.join(test_dir, "PNEUMONIA", "person1_virus_6.jpeg")
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

explainer = GradCAM()
grid = explainer.explain(
    validation_data=(img_array, np.array([1])),
    model=model,
    class_index=0,
    layer_name="conv5_block3_out"
)

plt.imshow(grid)
plt.axis('off')
plt.title("Grad-CAM for Pneumonia Detection")
plt.show()