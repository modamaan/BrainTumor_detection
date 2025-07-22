import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = glob.glob("runs/detect/predict/*.jpg")
images_to_display = images[0:10]

fig, axes = plt.subplots(2, 5, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if i < len(images_to_display):
        img = mpimg.imread(images_to_display[i])
        ax.imshow(img)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
