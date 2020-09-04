from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
plt.figure('pokemon')
img = Image.open('/Users/zuobangbang/Desktop/1.jpeg')
# plt.imshow(img)
# plt.show()
height,width=img.size
gray=img.convert('L')#*************************************************convert()见下文
# plt.imshow(gray.convert('RGB'),cmap ='gray')
data = gray.getdata()
# plt.imshow(gray)
# plt.show()
print(height)
data = np.mat(data,dtype='float')
print(height)
new_data = np.reshape(data,(width,height))
# print(new_data)
z6=[]
for i in range(1228,1263):
    q=[]
    for j in range(449,470):
        q.append(int(new_data[i,j]))
    z6.append(q)
print(z6)
z0=[]
for i in range(1228,1263):
    q=[]
    for j in range(352,370):
        q.append(int(new_data[i,j]))
    z0.append(q)
z5=[]
for i in range(1228,1263):
    q=[]
    for j in range(374,395):
        q.append(int(new_data[i,j]))
    z5.append(q)
m,n=new_data.shape[0],new_data.shape[1]
# z=[]
for i in range(1228,1263):
    q=[]
    for j in range(374,395):
        new_data[i,j]=z6[i-1228][j-374]
    for j in range(431,449):
        new_data[i,j]=z0[i-1228][j-431]
    for j in range(449,470):
        new_data[i,j]=z5[i-1228][j-449]
#     z.append(q)
# z=np.mat(z)
# print(z)
newp=Image.fromarray(new_data)
print(newp)
r=newp.convert(mode='RGB')
r.save('/Users/zuobangbang/Desktop/2.jpeg')
plt.imshow(r)

plt.axis('off')

plt.show()