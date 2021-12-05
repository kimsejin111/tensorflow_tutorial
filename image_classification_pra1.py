import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Error 메세지 지우는 코드
import PIL
import time
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image

'''이미지 받아오기'''
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir) # 무엇을 의미하는 코디이지?


'''이미지 로더 크기 정의'''
batch_size =32
img_height=180
img_width=180

'''저장된 이미지 갯수 출력'''
image_count = len(list(data_dir.glob('*/*.jpg'))) #파일 확장자로 이미지 갯수 확인
print(image_count) #이미지 갯수 출력

'''이미지 열기'''
#roses = list(data_dir.glob('roses/*'))  #파일 명으로 장미 분류
#Image.open(str(roses[0]))


'''이미지 열기'''
#tulips = list(data_dir.glob('tulips/*')) #파일 명으로 튤립 분류
#x=Image.open(str(tulips[0])) #이미지 출력을 위한 변수 저장
#x.show() # 이미지 파일 열기

'''훈련 데이트세트 정의'''
train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # 왜 0.2로 하는거지?
    subset="training",
    seed=123,   # seed는 무얼 의미하는 거지?
    image_size=(img_height, img_width),
    batch_size=batch_size
)

'''검증 데이트세트 정의'''
val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

'''class_names 속성에서 클래스 이름 찾기'''
class_names = train_ds.class_names
#print(class_names)

'''데이터 시각화 하기'''
f1=plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1): #take 함수는 무슨 함수일까
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8")) #numpy,imshow 는 무슨 함수일까
        plt.title(class_names[labels[i]])
        plt.axis("off")

#f1.savefig('default.png') #이미지 저장 함수 사용


'''이미지 집단/ 라벨의 배열 정보 확인'''
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


'''성능 최적화를 위해 cashe()와 prefetch() 이용하여 데이터세트 구성'''
AUTOTUNE =tf.data.experimental.AUTOTUNE
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE)


"데이터 세트 표준화-> RGB 값을 0~255 --> 0~1로 표준화"
normaliztion_layer= layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds=train_ds.map(lambda x,y:(normaliztion_layer(x),y))
image_batch,labels_batch = next(iter(normalized_ds)) #자동 반복 함수
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

'''모델 만들기
   Covolution 2D block 3ea
   relu: 활성화 함수, sigmoid의 기울기 소슬 문제 해결 --> 유튜브로 Relu 함수 강의 찾아보기
'''
num_classes=5
model=Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height,img_width,3)),
        layers.Conv2D(16,3,padding='same',activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'),
        layers.MaxPool2D(), #Pooling 필터링 개념으로 생각하자  해당 convolution에서 MAX값으로 필터링 하여 데이터 크기를 줄임
        layers.Flatten(), #convolution의 dimension을 줄여주는 함수
        layers.Dense(128,activation='relu'),
        layers.Dense(num_classes)
    ]
)
'''Optimizer & Loss 함수 설정 훈련 정확성을 알기 위해 metircs 설정'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
#model.summary()


'''Model training'''
epochs=10
history=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

'''Training result visualization'''
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range= range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label="Training ACC")
plt.plot(epochs_range,val_acc,label="Validation ACC")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label="Training Loss")
plt.plot(epochs_range,val_loss,label="Validation Loss")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

#git test 확인

