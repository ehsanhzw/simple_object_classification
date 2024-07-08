from nn_models import *
from keras import datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import os

model_name = 'slowunet' # cnn / unet / quickcnn / param / slowunet #
chosen_classes = ['Cat','Dog'] # Choose two
epochs = 20

model_dir = os.path.join('.','models',model_name)

(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()
train_images,test_images=train_images/255,test_images/255
classes_labels = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

## only cat/dog datasets are selected:
train_filter = np.where((train_labels == classes_labels.index(chosen_classes[0])) | (train_labels == classes_labels.index(chosen_classes[1])))[0]
train_images = train_images[train_filter]
train_labels = train_labels[train_filter]
print('shape of train images dataset: ',np.shape(train_images))
test_filter = np.where((test_labels == classes_labels.index(chosen_classes[0])) | (test_labels == classes_labels.index(chosen_classes[1])))[0]
test_images = test_images[test_filter]
test_labels = test_labels[test_filter]
print('shape of train images dataset: ', np.shape(test_images))

## Uncomment if you want to view first samples of selected dataset
# for i in range(12):
#     plt.subplot(3,4,i+1)
#     plt.imshow(train_images[i])
#     plt.xticks([]); plt.yticks([])
#     plt.xlabel(classes_labels[train_labels[i][0]])
# plt.show()

input_shape = (32,32,3)
if model_name == 'cnn':
    model = cnn_model(input_shape)
elif model_name == 'unet':
    model = unet_model(input_shape)
elif model_name == 'slowunet':
    model = slowunet_model(input_shape)
elif model_name == 'quickcnn':
    model = quickcnn_model(input_shape)
elif model_name == 'param':
    model = param_model(input_shape)
else:
    pass # will raise error if given wrong model name

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images,train_labels,epochs=epochs,validation_data=(test_images,test_labels))

utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=os.path.join(model_dir,model_name)+'_model.png')
print(model.summary())

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.suptitle(model_name.upper()+' Model Accuracy/Loss Per Epoch')
fig.set_size_inches(13.6, 7.6)
plt.savefig(os.path.join(model_dir,model_name)+'_AccuracyLoss_plot.png',dpi=141)
plt.show()

loss,accuracy = model.evaluate(test_images,test_labels)
f = open(os.path.join(model_dir,model_name)+'_metrics_summary.txt', 'w')
f.write(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
f.close()
model.save(os.path.join(model_dir,model_name)+'_model.keras')