#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
plt.style.use("ggplot")

from skimage import io
from sklearn.cluster import KMeans

from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider
import ipywidgets as widgets


# In[ ]:


plt.rcParams['figure.figsize'] = (20, 12)


# In[ ]:


img = io.imread('images/1-Saint-Basils-Cathedral.jpg')
ax = plt.axes(xticks=[],yticks=[])
ax.imshow(img);


# In[ ]:


img.shape


# In[ ]:


img_data = (img/255.0).reshape(-1,3)
img_data.shape


# In[ ]:


from plot_utils import plot_utils


# In[ ]:


x = plot_utils(img_data, title="input color space:over 16 million possible colors")
x.colorSpace()


# In[ ]:


from sklearn.cluster import MiniBatchKMeans


# In[ ]:


kmeans = MiniBatchKMeans(16).fit(img_data)
kcolors = kmeans.cluster_centers_[kmeans.predict(img_data)]
y = plot_utils(img_data,colors = kcolors, title="Reduced color space:over 16 colors")
y.colorSpace()


# In[ ]:


img_dir = 'images/'


# In[ ]:


@interact
def color_compression(image = os.listdir(imd_dir),k = IntSlider(min = 1,max = 256,step = 1,value = 16,continuous_update=False,layout = dict(width = '100%')))
input_img = io.imread(img_dir+image)
img_data = (input_img/255.0).reshape(-1,3)

kmeans = MiniBatchKMeans(k).fit(img_data)
kcolors = kmeans.cluster_centers_[kmeans.predict(img_data)]
k_img = np.reshape(k_colors,(input_img.shape))


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle('K-Means image compression',fontsize = 20)

ax1.set_title('compressed image')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(k_img)

ax2.set_title('original (16,777,216 colors)')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(img_data)

plt.subplot_adjust(top=0.85)
plt.show()


# In[ ]:




