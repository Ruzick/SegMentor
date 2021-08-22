import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image, ImageOps
from matplotlib import gridspec


def read_image(image):
    """
    reads image and transfomrs it into numpy array 
    Arguments:
    imgage [jpg]: image file
    Results 
    [numpy array] of dimensions (m,n,3) for color (m,n)for gray
    """
    return mpimg.imread(image)
    
def format_image(image):
    """
    converts array to a Pillow image
    Arguments:
    image [np array] : dimensions (m,n,3) for color, (m,n) grayscale
    Results
    Pillow image object
    """
    image = Image.fromarray(image)
    return image 

def get_category(img,model):
    """Write a Function to Predict the Class Name
    Args:
        img [jpg]: image file
        model[tflite]
    Returns:
        [str]: Prediction
    """
    #Prepare iage further for running inference *******
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/model/')

    tflite_model_file = model
    # Load TFLite model and allocate tensors.
    with open(path + tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # Interpreter interface for TensorFlow Lite Models.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)

    # Gets model input and output details.
    input_index = interpreter.get_input_details()[0]["index"]
    input_details = interpreter.get_input_details()
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.allocate_tensors()
    #read image and make it into pillow object
    input_img = read_image(img)
    image = format_image(input_img)

    # Get image size - Getting the (width, height) the model needs from the input image -converting from BHWC to WH
    input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
    old_size = image.size  # old_size is in (width, height) format
    desired_ratio = input_size[0] / input_size[1]
    old_ratio = old_size[0] / old_size[1]

    if old_ratio < desired_ratio: # '<': cropping, '>': padding
    #if the ratio of the downloaded image is less than the one of the ideal input, then the width of the downloaded is too high
    #change size to match desired ratio
        new_size = (old_size[0], int(old_size[0] / desired_ratio))
    else:
        new_size = (int(old_size[1] * desired_ratio), old_size[1])


    # Cropping the original image to the desired aspect ratio
    delta_w = new_size[0] - old_size[0]
    delta_h = new_size[1] - old_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    cropped_image = ImageOps.expand(image, padding)

    # Resize the cropped image to the desired model size
    resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)

    # Convert to a NumPy array, add a batch dimension, and normalize the image.
    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)
    image_for_prediction = image_for_prediction / 127.5 - 1

    # Invoke the interpreter to run inference.
    # Interpreter interface for TensorFlow Lite Models.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
      # Sets the value of the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    # Invoke the interpreter.
    interpreter.invoke()

    #get values of input sizes **********
    input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
    predictions_array = interpreter.get_tensor(output_index)
    raw_prediction = predictions_array
    # Post-processing: convert raw output to segmentation output
    ## Method 1: argmax before resize - this is used in some frozen graph
    # seg_map = np.squeeze(np.argmax(raw_prediction, axis=3)).astype(np.int8)
    # seg_map = np.asarray(Image.fromarray(seg_map).resize(image.size, resample=Image.NEAREST))
    ## Method 2: resize then argmax - this is used in some other frozen graph and produce smoother output
    width, height = cropped_image.size
    seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

    
    def create_pascal_label_colormap():
    #"""Creates a label colormap used in PASCAL VOC segmentation benchmark. Returns:A Colormap for visualizing segmentation results."""
        colormap = np.zeros((257, 3), dtype=int)#np.zeros((256, 3), dtype=int)
        ind = np.arange(257, dtype=int)# np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3): 
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        return colormap


    def label_to_color_image(label):
        """Adds color defined by the dataset colormap to the label.

        Args:
        label: A 2D array with integer type, storing the segmentation label.

        Returns:
            result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

        Raises:
            ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        colormap = create_pascal_label_colormap()
        if np.max(label) >= len(colormap):
            raise ValueError('label value too large.')

        return colormap[label]

    def vis_segmentation(image, seg_map,model):
        """
        Visualizes input image, segmentation map and overlay view
        Arguments:
        image [Pil image] : (m,n)
        seg_map [np_array] (m,n)
        model [tflite]: file
        Results
        An image[png] saved on path 'static/images/'
        """
        plt.figure(figsize=(15, 5))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

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
        plt.imshow(
            FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')
        # plt.show()

  
        path_pic = os.path.join(ROOT_DIR + '/static/images/')
        fig = plt.gcf()
        # fig.savefig(f'static/images/pic_{model}.png')
        fig.savefig(path_pic+f'pic_{model}.png')
        # import io
        # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        # output = io.BytesIO()
        # # FigureCanvas(fig).print_png(output)
        # FigureCanvas(fig).print_png(output)
        # return output
        #End of Vis_seg function

    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
        # predicted_label = np.argmax(predictions_array)
        # #Prediction: (1, 257, 257, 21)
    
    im_output = vis_segmentation(cropped_image, seg_map, model=model) 

    return im_output 

def save_image(img, img_name):
    """Saves input image
    Args:
        img [jpg]: image file
        img_name[string]: name for the file
    Results:
        [png] saves image to /static/images/
    """
    read_img = mpimg.imread(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR + f'/static/images/{img_name}.png')
    print(file_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    plt.imsave(file_path, read_img)

