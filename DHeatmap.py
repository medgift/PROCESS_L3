
# coding: utf-8

# In[1]:


from openslide import OpenSlide
from os import listdir
from os.path import join, isfile, exists, splitext
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util import otsu_thresholding, center_of_slide_level,connected_component_image
from skimage import measure
from scipy import ndimage
from functions import *
from models import getModel
import matplotlib.pyplot as plt
import time
from skimage.viewer import ImageViewer
import tensorflow as tf
import keras
import warnings
warnings.filterwarnings("ignore")   
from scipy import interpolate

files =os.listdir("/mnt/nas4/datasets/ToReadme/CAMELYON17/lesion_annotations")
print files[10][:-4]


# In[1]:


import tensorflow as tf
tf.__version__


# In[2]:


import setproctitle
EXPERIMENT_TYPE = 'distributed_inference'
# SET PROCESS TITLE
setproctitle.setproctitle('UC1_{}_{}'.format(EXPERIMENT_TYPE, 'mara'))


# In[67]:


pwd=""
from os.path import join,isfile, exists, splitext
def get_folder(file_name, path="/mnt/nas4/datasets/ToReadme/CAMELYON17/"):
    for fold in os.listdir(path)[:4]:
        m = file_name+".tif"
        if m in os.listdir(path+fold):
            return fold
    return path
def preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1):
    rgb_im, slide = load_slide(slide_path, slide_level=slide_level)
    otsu_im = get_otsu_im(rgb_im, verbose = 0)
    return slide, rgb_im, otsu_im

def apply_morph(otsu_im):
    im_gray_ostu = otsu_im
    kernel = np.ones((2,2),np.uint8)
    kernel_1 = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel)
    opening_1 = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel_1)
    closing = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_CLOSE,kernel)
    opening_1= np.abs(255-opening_1)
    return opening_1

def preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1):
    rgb_im, slide = load_slide(slide_path, slide_level=slide_level)
    otsu_im = get_otsu_im(rgb_im, verbose = 0)
    return slide, rgb_im, otsu_im

def apply_morph(otsu_im):
    im_gray_ostu = otsu_im
    kernel = np.ones((2,2),np.uint8)
    kernel_1 = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel)
    opening_1 = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel_1)
    closing = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_CLOSE,kernel)
    opening_1= np.abs(255-opening_1)
    return opening_1

training_slide=True
file_name='patient_051_node_2'
xml_path="/mnt/nas4/datasets/ToReadme/CAMELYON17/lesion_annotations/"+file_name+'.xml'
folder = get_folder(file_name)
filename ="/mnt/nas4/datasets/ToReadme/CAMELYON17/"+folder+"/"+file_name+'.tif' 
slide_path = join(pwd,filename)
print "file name : "+slide_path+"\n"
if isfile(slide_path):
    """is it file? """
    slide=OpenSlide(slide_path)
elif exists(slide_path):
    """ dose it exist? """
    print "slide_path :" + slide_path + " is not a readable file \n"
else:
    """ it is not a file and doesn't exist"""
    print "file dosen't exist in this path :"  + slide_path+"\n"

slide_w, slide_h = slide.dimensions
print "Whole Slide dimensions (with, heigth):{0}\n".format(slide.dimensions)
#Slide.level_dimensions
slide_level = 7
s_level_w, s_level_h = slide.level_dimensions[slide_level]
print "slide.level_count-1 dimensions (width,heigth):{0}\n".format(slide.level_dimensions[slide_level])
slide, rgb_im, otsu_im = preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1)

if not training_slide:
    slide, rgb_im, otsu_im = preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1)

    ## to be continued....
else:
    slide, annotations_mask, rgb_im, im_contour = preprocess(slide_path, xml_path, slide_level=slide_level)


# In[68]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = sys.argv[2] 
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

folder='./cam1617_2009/'
#0609-1648/'
CONFIG_FILE=folder+'config.cfg'

settings = parseTrainingOptions(CONFIG_FILE)
print settings

model=getModel(settings)
model.load_weights(folder+'tumor_classifier.h5')


# In[69]:


dmodels={}


# In[70]:


import subprocess as sp
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values

free_gpu_memory_ = get_gpu_memory()
MEMO_REQUIREMENT = 3000.
n_models = int(free_gpu_memory_[0] / MEMO_REQUIREMENT)
# Distribute inference : GPU Parallelism
#while True:
#    i=1
print "Distributing inference over {} model copies".format(n_models)
for i in range(0, n_models):
    try: 
        print "Instantiating model n. ", i
        nmodel=getModel(settings)
        print "Loading weights..."
        nmodel.load_weights(folder+'tumor_classifier.h5')
        print "Adding model to available models."
        dmodels[i]=nmodel
        i+=1
    except:
        print "Something went wrong: MEMO_REQUIREMENT TOO LITTLE"
        break


# In[71]:


import multiprocessing
opening_1 = apply_morph(otsu_im)
plt.rcParams['figure.figsize']=(5,5)
y_low, x = np.unique(np.where(opening_1>0)[0]), np.unique(np.where(opening_1>0)[1]) 
patch_size=224
patches=[]
flag=False
#sebas: here you can leave the code as it is
print 'Heatmap dimensions: ', slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]


# In[72]:


heatmap=np.zeros((slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]))
seen=np.zeros((slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]))        
resolution=2
mesh = np.meshgrid(np.arange(0, slide.level_dimensions[slide_level][0], 2),np.arange(0, slide.level_dimensions[slide_level][1], 2))
positions = np.vstack(map(np.ravel, mesh)).T
final_p=[]
for p in positions:
    x,y=p[0],p[1]
    if np.sum(opening_1[y-(resolution/2):y+(resolution/2), x-(resolution/2):x+(resolution/2)])>0:
        final_p.append(p)
n_models=1        
def worker(slide, locations_vector, locations_index, data_batch, data_locations, batch_size=32):
    """worker function for multiprocessing"""
    batch=[]
    batch_locations = locations_vector[locations_index.value:locations_index.value+batch_size]
    for l in batch_locations:
        #l[0] is x, l[1] is y
        patch=np.asarray(slide.read_region((l[0]*128,l[1]*128),0,(224,224)))[...,:3]
        batch.append(np.asarray(patch)[...,:3])
        #Image.fromarray(patch).save('prova_batch/{}-{}.png'.format(l[1],l[0]))
    data_batch[0]=batch
    data_locations[0]=batch_locations
    with locations_index.get_lock():
        locations_index.value +=batch_size
    return

start_time = time.time()
locations_index = multiprocessing.Value("i", 0)
manager = multiprocessing.Manager()
batches = {}
locations = {}
for b in range(n_models):
    batches[b]=manager.dict()
    locations[b]=manager.dict()
#batches=manager.dict()
batch_size=32
while locations_index.value < len(final_p):
    jobs = []
    for m in range(n_models):
        p = multiprocessing.Process(target=worker, 
                                    args=(slide, 
                                          final_p, 
                                          locations_index, 
                                          batches[m], 
                                          locations[m]))
        jobs.append(p)
        p.start() 
        p.join()
        predictions=dmodels[m].predict(np.reshape(batches[m][0],(len(batches[m][0]),224,224,3)))
        for p in range(len(predictions)):
            x_b, y_b=locations[m][0][p][0], locations[m][0][p][1]
            heatmap[y_b, x_b]=predictions[p][0]
            seen[y_b,x_b]=1
end_time = time.time()

points = np.asarray(seen.nonzero()).T
values = heatmap[heatmap.nonzero()]

grid_x, grid_y = np.mgrid[0:slide.level_dimensions[slide_level][1]:1, 
                          0:slide.level_dimensions[slide_level][0]:1]
interpolated_heatmap = interpolate.griddata(points, values,
                                      (grid_x, grid_y), 
                                        fill_value=0.
                                       )
print 'Number of patches analysed: ', np.sum(seen)
print 'Elapsed time: ', end_time-start_time
plt.rcParams['figure.figsize']=(25,25)
plt.imshow(im_contour)
#plt.imshow(heatmap, alpha=0.5)
plt.imshow(interpolated_heatmap, alpha=0.5)


# In[73]:


interpolated_heatmap>0.5


# In[74]:


plt.imshow(im_contour)
#plt.imshow(heatmap, alpha=0.5)
plt.imshow(interpolated_heatmap[np.argwhere(interpolated_heatmap>0.5)], alpha=0.5, cmap='jet')


# In[ ]:


interpolated_heatmap = interpolate.griddata(points, values,
                                      (grid_x, grid_y), 
                                        fill_value=1.
                                       )
print 'Number of patches analysed: ', np.sum(seen)
print 'Elapsed time: ', end_time-start_time
plt.rcParams['figure.figsize']=(25,25)
plt.imshow(im_contour)
#plt.imshow(heatmap, alpha=0.5)
plt.imshow(heatmap, cmap="jet", alpha=0.5)


# In[ ]:


plt.imshow(heatmap)


# In[ ]:


resolution 


# In[ ]:



points = np.asarray(seen.nonzero()).T
values = heatmap[heatmap.nonzero()]

grid_x, grid_y = np.mgrid[0:slide.level_dimensions[slide_level][1]:1, 
                          0:slide.level_dimensions[slide_level][0]:1]
interpolated_heatmap = interpolate.griddata(points, values,
                                      (grid_x, grid_y), 
                                        fill_value=0.
                                       )
print 'Number of patches analysed: ', np.sum(seen)
print 'Elapsed time: ', end_time-start_time
plt.rcParams['figure.figsize']=(25,25)
plt.imshow(im_contour)
#plt.imshow(heatmap, alpha=0.5)
plt.imshow(interpolated_heatmap, alpha=0.5)


# In[ ]:


interpolated_heatmap[np.argwhere(interpolated_heatmap>0.5)].shape()


# In[ ]:


interpolated_heatmap.shape


# In[ ]:


th=0.9


# In[ ]:


thr_heatmap=interpolated_heatmap[np.argwhere(interpolated_heatmap>th).T[0], np.argwhere(interpolated_heatmap>th).T[1]]


# In[ ]:


thr_heat=np.zeros((slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]))


# In[ ]:


thr_heat[np.argwhere(interpolated_heatmap>th).T[0], np.argwhere(interpolated_heatmap>th).T[1]]=interpolated_heatmap[np.argwhere(interpolated_heatmap>th).T[0], np.argwhere(interpolated_heatmap>th).T[1]]


# In[ ]:


plt.rcParams['figure.figsize']=(25,25)
plt.imshow(im_contour)
plt.imshow(thr_heat, alpha=0.5, cmap='jet')

