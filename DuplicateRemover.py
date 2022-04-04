from PIL import Image
import imagehash
import os
import numpy as np
import pandas as pd

# example of using the vgg16 model as a feature extraction model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump 
from sklearn.cluster import KMeans

class DuplicateRemover:
    def __init__(self,dirname,hash_size = 8):
        self.dirname = dirname
        self.hash_size = hash_size
    
    def vgg_vectors(self, input_image):
        # load an image from file
        image = load_img(input_image, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # load model
        model = VGG16()
        # remove the output layer
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # get extracted features
        features = model.predict(image)
        # print(features.shape)
        return features
        # save to file
        # dump(features, open('dog.pkl', 'wb'))

    # def vgg_find_similar(self, given_image):
    #     fnames = os.listdir(self.dirname)
    #     given_image_vector = self.vgg_vectors(given_image)
    #     all_image_vectors = []
    #     for image in fnames:
    #         all_image_vectors.append(vgg_vectors(image))

    def vgg_all_clusters(self):
        fnames = os.listdir(self.dirname)
        vector_length = 4096
        embeddings = []
        number_of_images = len(fnames)
        for image in fnames:
            embeddings.append(self.vgg_vectors(os.path.join(self.dirname,image)))
        embeddings_new = np.asarray(embeddings, dtype=np.float32).reshape((number_of_images, vector_length))
        kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings_new)

        print(fnames)
        print(kmeans.labels_)
        result = {}
        for idx,label in enumerate(kmeans.labels_):
            if(label in result):
                result[label].append(fnames[idx])
            else:
                result[label] = [fnames[idx]]
        print(result)

    def ccd_vgg_all_clusters(self, urns):
        vector_length = 4096
        embeddings = []
        number_of_images = len(urns)
        for image in urns:
            embeddings.append(self.vgg_vectors(os.path.join(self.dirname,image)))
        embeddings_new = np.asarray(embeddings, dtype=np.float32).reshape((number_of_images, vector_length))
        kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings_new)

        print(urns)
        print(kmeans.labels_)
        result = {}
        for idx,label in enumerate(kmeans.labels_):
            if(label in result):
                result[label].append(urns[idx])
            else:
                result[label] = [urns[idx]]
        
        flat_list = [item for sublist in list(result.values()) for item in sublist]
        stringList = flat_list[0]
        for urn in flat_list[1:]:
            stringList += "#"+urn
        return stringList
