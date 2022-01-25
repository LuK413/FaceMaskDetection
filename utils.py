import os
import xml.etree.ElementTree as ET
import cv2
from matplotlib import pyplot as plt
import shutil
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Globals
IMG_WIDTH = 640
IMG_HEIGHT = 480
inv_names = {
    '0': 'with_mask',
    '1': 'mask_weared_incorrect',
    '2': 'without_mask'
}
cmap = {
    'without_mask': (255, 0, 0),
    'with_mask': (0, 255, 0),
    'mask_weared_incorrect': (255, 255, 0)
}
names = {
    'with_mask': 0,
    'mask_weared_incorrect': 1,
    'without_mask': 2
}


def parse_xml(filename):
    dirname = os.path.abspath('')
    absolute_path = os.path.join(dirname, filename)
    tree = ET.parse(absolute_path)
    root = tree.getroot()

    height = int(root.find('size').findtext('height'))
    width = int(root.find('size').findtext('width'))
    name = root.findtext('filename') # need this?
    boxes = []

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        # cap the bounding box within the image so that bounding boxes are not outside the image
        xmin = max(0, int(bndbox.findtext('xmin')))
        ymin = max(0, int(bndbox.findtext('ymin')))
        xmax = min(width, int(bndbox.findtext('xmax')))
        ymax = min(height, int(bndbox.findtext('ymax')))
        mask = obj.findtext('name')
        boxes.append([xmin, ymin, xmax, ymax, mask])

    return boxes, (height, width), name


def parse_yolo(filename):
    dirname = os.path.abspath('')
    absolute_path = os.path.join(dirname, filename)
    with open(filename) as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        annot = line.split(' ')

        box_width = int(float(annot[3]) * IMG_WIDTH)
        box_height = int(float(annot[4]) * IMG_HEIGHT)
        xmin = int(float(annot[1]) * IMG_WIDTH - (box_width / 2))
        ymin = int(float(annot[2]) * IMG_HEIGHT - (box_height / 2))
        xmax = int(float(annot[1]) * IMG_WIDTH + (box_width / 2))
        ymax = int(float(annot[2]) * IMG_HEIGHT + (box_height / 2))

        boxes.append([xmin, ymin, xmax, ymax, inv_names[annot[0]]])

    return boxes


def view_annotations(filename):
    dirname = os.path.abspath('')
    absolute_path = os.path.join(dirname, filename)
    bgr_img = cv2.imread(absolute_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    name = filename.split('/')[-1][:-3]

    img_path = 'datasets/face-mask-detection/annotations/' + name + 'xml'

    if 'resized' in filename:
        annotations = parse_yolo('datasets/face-mask-detection/labels/' + name + 'txt')
    else:
        annotations, _, _ = parse_xml(img_path)

    for annot in annotations:
        start_point = (annot[0], annot[1])
        end_point = (annot[2], annot[3])
        mask_status = annot[4]
        color = cmap[mask_status]
        cv2.rectangle(rgb_img, start_point, end_point, color, 2)
        cv2.putText(rgb_img, mask_status, (annot[0] - 20, annot[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 1,
                    cv2.LINE_AA)
    plt.imshow(rgb_img)


def display_img(filename):
    img = cv2.imread(filename)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb_img)


def convert_to_yolov5_format(bndbox):
    xmin, ymin, xmax, ymax, name = bndbox
    x_center = ((xmin + xmax) / 2) / IMG_WIDTH
    y_center = ((ymin + ymax) / 2) / IMG_HEIGHT
    normalized_width = (xmax - xmin) / IMG_WIDTH
    normalized_height = (ymax - ymin) / IMG_HEIGHT
    return name, x_center, y_center, normalized_width, normalized_height


def move_files(lof, dest):
    try:
        for file in lof:
            # resize first, then copy
            shutil.copy(file, dest)
    except shutil.Error as e:
        print('Please check that the files and destination folders are correct.')


def plot_results(x, y):
    fig = make_subplots(rows=7, cols=2, vertical_spacing=0.075, subplot_titles=tuple(y.columns))
    for index, col in enumerate(y):
        row_num = index // 2 + 1
        col_num = (index % 2) + 1
        fig.add_trace(
            go.Scatter(x=x, y=y[col], name=col),
            row=row_num, col=col_num
        )
        fig.update_xaxes(title_text='epochs', row=row_num, col=col_num)
        fig.update_yaxes(title_text=col, row=row_num, col=col_num)
    fig.update_layout(height=1200)
    fig.show()


def plot_loss(x, y, loss_type):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y['train/'+loss_type], name='train/'+loss_type)
    )
    fig.add_trace(
        go.Scatter(x=x, y=y['val/'+loss_type], name='val/'+loss_type)
    )
    fig.update_xaxes(title_text='epochs')
    fig.update_yaxes(title_text=loss_type)
    fig.update_layout(title={
        'text': loss_type,
        'x': 0.5
    })
    fig.show()