from keras import models
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

images_path = os.path.join('.','Images','CD')
model_name = 'unet' # cnn / unet / quickcnn / param #
model_dir = os.path.join('.','models',model_name)
chosen_classes = ['Cat','Dog']

classes_labels = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

model = models.load_model(os.path.join(model_dir,model_name)+'_model.keras')
print(model.summary())

JPGs = os.listdir(images_path)
Img = []
for i in range(len(JPGs)):
    Img.append(cv2.imread(os.path.join(images_path,JPGs[i])))
    Img[i] = cv2.cvtColor(Img[i],cv2.COLOR_BGR2RGB)
    Img[i] = (cv2.resize(Img[i],(32,32)))/255
fig = plt.figure()
for i in range(len(Img)):
    plt.subplot(5,6,i+1)
    plt.imshow(Img[i])
    plt.xticks([]); plt.yticks([])
    certainty, prediction = max(model.predict(np.array([Img[i]]))[0]),np.argmax(model.predict(np.array([Img[i]]))[0])
    plt.xlabel(classes_labels[prediction] + f' %{certainty:.2f}')
fig.set_size_inches(13.5966,  7.6481)
plt.suptitle(model_name.upper() + ' Manual Test')
plt.savefig(os.path.join(model_dir,model_name)+'_'+chosen_classes[0]+chosen_classes[1]+'_validation.png',dpi=141.2120)
plt.show()
