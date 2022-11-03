 
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# Create your views here.
from keras.models import load_model
from keras.preprocessing import image
import json

img_height, img_width = 128, 128

with open('./model/label.json', 'r') as f:
    labelinfo = f.read()

labelInfo = json.loads(labelinfo)
model = load_model('./model/vgg16.h5')


def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)


def predictImage(request):

    fileObj = request.FILES['filepath']
    fs = FileSystemStorage()
    fs.save(fileObj.name, fileObj)
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1, img_height, img_width, 3)
    predi = model.predict(x)

    import numpy as np
    predictedLabel = labelInfo[str(np.argmax(predi))]

    context = {'filePathName': filePathName,
               'predictedLabel': predictedLabel[0]}
    return render(request, 'index.html', context)
