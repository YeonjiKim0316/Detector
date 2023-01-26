import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np

st.title('마스크 디텍션 모델')

flag = False
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('./keras_Model.h5', compile=False)

# Load the labels
class_names = open('./labels.txt', 'r').readlines()

img_file_buffer = st.camera_input("화면 중앙에 얼굴을 위치시켜 주세요")


if img_file_buffer is not None:
    flag = True

if flag == True:
# To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # # Replace this with the path to your image
    # image = Image.open('<IMAGE_PATH>').convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    width, height = img.size   # Get dimensions
    new_width = 224
    new_height = 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(img)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    st.write('현재 상태:', class_name)
    st.write('Confidence score:', confidence_score)
