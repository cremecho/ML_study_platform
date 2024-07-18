import json
import glob
import base64
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


file_ls = glob.glob("*.json")
data_dict = {}
for file in file_ls:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        image_data = data['imageData']
        image_data = base64.b64decode(image_data)
        img = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        label = data['shapes'][0]['points']

        file_data = {"image":img, "label":label}
        data_dict[data["imagePath"]] = file_data



from sklearn.decomposition import PCA
from PIL import Image


im_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
pca = PCA(20)
pca.fit(im_gray)
fp = pca.components_[1,:]
fp = fp.squeeze()
recon = np.dot(im_gray,fp)


pca.fit(np.transpose(im_gray))
recon2 = np.dot(pca.components_[1,:].squeeze(), im_gray)

# 创建包含两个子图的图形
x = range(len(recon))
x2 = range(len(recon2))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# 在第一个子图中显示原始图像
ax1.imshow(img)
ax1.set_title('Original Image')
ax1.axis('off')

# 在第二个子图中显示重建结果
ax2.plot(x, recon)
ax2.set_title('PCA Reconstruction (vertical)')
ax2.set_xlabel('Index')
ax2.set_ylabel('Reconstructed Value')


ax3.plot(x2, recon2)
ax3.set_title('PCA Reconstruction (horizental)')
ax3.set_xlabel('Index')
ax3.set_ylabel('Reconstructed Value')

# 显示图形
plt.show()