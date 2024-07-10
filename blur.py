import os
import tarfile
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from six.moves import urllib
import matplotlib.pyplot as plt
import tempfile
import urllib.request

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None

        # Extract frozen graph from tar archive
        with tarfile.open(tarball_path, 'r') as tar_file:
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                    break

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image."""
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, resample=Image.LANCZOS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
        )
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark."""
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label."""
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = plt.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
model_path = os.path.join(model_dir, _TARBALL_NAME)

if not os.path.exists(model_path):
    # Model tarball does not exist, so download it
    print('Downloading model, this might take a while...')
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], model_path)
    print('Download completed!')
else:
    print('Model already exists locally.')

# Proceed with loading the DeepLab model from the downloaded tarball
# Assuming DeepLabModel is a class defined elsewhere in your code
MODEL = DeepLabModel(model_path)
print('Model loaded successfully!')

def run_visualization(image_path):
    """Inferences DeepLab model and visualizes result."""
    try:
        original_im = Image.open(image_path)
    except IOError:
        print('Cannot retrieve image. Please check path:', image_path)
        return None, None

    print('running deeplab on image')
    resized_im, seg_map = MODEL.run(original_im)
    vis_segmentation(resized_im, seg_map)
    return resized_im, seg_map

IMAGE_NAME = 'm&c.jpg'

def save_unique_values(seg_map):
    global unique_values
    unique_values = np.unique(seg_map)
    print("Unique values in segmentation map:", unique_values)

resized_im, seg_map = run_visualization(IMAGE_NAME)
if resized_im is None or seg_map is None:
    raise RuntimeError("Failed to run visualization on the image")

save_unique_values(seg_map)

numpy_image = np.array(resized_im)

masks = {}
for i, value in enumerate(unique_values):
    if i == 0:
        person_not_person_mapping = numpy_image.copy()
        person_not_person_mapping[seg_map == 0] = 0
    else:
        person_not_person_mapping = numpy_image.copy()
        person_not_person_mapping[seg_map != value] = 0

plt.imshow(person_not_person_mapping)

original_image = Image.open(IMAGE_NAME)
original_image = np.array(original_image)

mapping_resized = cv2.resize(person_not_person_mapping, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
plt.imshow(mapping_resized)

gray = cv2.cvtColor(mapping_resized, cv2.COLOR_RGB2GRAY)
blurred_images = []
for size in unique_values:
    if size % 2 == 0:
        size += 1  # Make size odd
    blurred = cv2.GaussianBlur(gray, (size, size), 0)
    blurred_images.append(blurred)

ret, thresholded_img = cv2.threshold(blurred_images[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(thresholded_img, cmap="gray")

mapping = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR555)
plt.imshow(mapping)

blurred_original_image = cv2.GaussianBlur(original_image, (251, 251), 0)
plt.imshow(blurred_original_image)

layered_image = np.where(mapping != (0, 0, 0), original_image, blurred_original_image)
plt.imshow(layered_image)

cv2.imshow('Layered Image', layered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

