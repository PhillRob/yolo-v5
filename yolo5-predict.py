#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import cv2
import tensorflow as tf
import numpy as np

import os
from PIL import Image
from tqdm import tqdm
#import geopandas as gpd
# import pandas as pd
import string as str
#from shapely.geometry import Polygon
from osgeo import osr, gdal
from PIL import Image
from PIL.TiffTags import TAGS
from os.path import exists
Image.MAX_IMAGE_PIXELS = 27102756380


def main():

  # import model
  model_path = 'pointnet2/best_saved_model'
  confidence = 0.2
  width, height = 800, 800
  model = tf.saved_model.load(model_path)

  # prediction functions
  def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

      # Arguments
          image     : The image to draw on.
          box       : A list of 4 elements (x1, y1, x2, y2).
          color     : The color of the box.
          thickness : The thickness of the lines to draw a box with.
      """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


  def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
      for x in range(0, image.shape[1], stepSize):
        # yield the current window
        yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


  def generate_crops(image_file, winW=512, winH=512, stepSize=512 // 2):
    image = cv2.imread(image_file)
    height = np.size(image, 0)
    width = np.size(image, 1)

    crops = []
    resized = image
    if width / winW > 5 or width / winH > 5:
      print("Warning! Huge image, may take 5 mins")
      winW, winH, stepSize = winW, winH, stepSize
    for (x, y, window) in sliding_window(resized, stepSize, windowSize=(winW, winH)):
      if window.shape[0] != winH or window.shape[1] != winW:
        continue
      crops.append([x, y, x + winW, y + winH])
    return crops


  def non_max_suppression(boxes, overlapThresh=0.4):
    boxes = np.asarray(boxes)
    if len(boxes) == 0:
      return []
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)
      suppress = [last]
      for pos in range(0, last):
        j = idxs[pos]
        xx1 = max(x1[i], x1[j])
        yy1 = max(y1[i], y1[j])
        xx2 = min(x2[i], x2[j])
        yy2 = min(y2[i], y2[j])
        w = max(0, xx2 - xx1 + 1)
        h = max(0, yy2 - yy1 + 1)
        overlap = float(w * h) / area[j]
        if overlap > overlapThresh:
          suppress.append(pos)
      idxs = np.delete(idxs, suppress)
    return boxes[pick]


  def rescale_bb_box(bb_box, target_img_size, raw_img_size):
    raw_x = raw_img_size[1]
    raw_y = raw_img_size[0]

    gain = min(raw_img_size[0] / target_img_size[0], raw_img_size[1] / target_img_size[1])  # gain  = old / new
    pad = (raw_img_size[1] - target_img_size[1] * gain) / 2, (
      raw_img_size[0] - target_img_size[0] * gain) / 2  # wh padding
    x1 = (bb_box[0] - pad[0]) / gain
    y1 = (bb_box[1] - pad[1]) / gain
    x2 = (bb_box[2] - pad[0]) / gain
    y2 = (bb_box[3] - pad[1]) / gain
    return [int(x1), int(y1), int(x2), int(y2)]


  def letterbox(im, width, height, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    new_shape = (width, height)
    if isinstance(new_shape, int):
      new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
      r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
      dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
      dw, dh = 0.0, 0.0
      new_unpad = (new_shape[1], new_shape[0])
      ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
      im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


  def transform_bbox(each_list, target_img_size, raw_img_size):
    final_list = [rescale_bb_box(each_value, target_img_size, raw_img_size) for each_value in each_list]

    return final_list


  def extract_boxes_confidences_classids(nms, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []
    boxes_list = nms[0].numpy()
    probs = nms[1].numpy()
    classes = nms[2].numpy()
    for each_box, score in zip(boxes_list[0], probs[0]):
      classID = 0
      conf = score
      if conf > confidence:
        w, h = each_box[2] - each_box[0], each_box[3] - each_box[1]
        x, y = (each_box[0] + each_box[2]) / 2, (each_box[1] + each_box[3]) / 2
        x, w, y, h = (x, w, y, h) * np.array([width, height, width, height])
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        confidences.append(float(conf))
        classIDs.append(int(classID))
    return boxes, confidences, classIDs


  def load_img(img_path, width, height):
    image = img_path
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, width, height, stride=32)
    image = image.astype(np.float32)
    image /= 255
    # height, width = raw_res
    image = np.ascontiguousarray(image)
    image = np.expand_dims(image, axis=0)
    return image


  def predict(model, img, width, height):
    image = load_img(img, width, height)
    infer = model.signatures["serving_default"]
    predictions = model(image)
    return predictions


  def get_predictions(img, confidence=.05, width=800, height=800):
    predictions = predict(model, img, width, height)
    boxes, confidences, classIDs = extract_boxes_confidences_classids(predictions, confidence, width, height)
    image_bbox_dtls = {}
    pred_img_size = (width, height)
    target_image = img
    target_img_size = target_image.shape[0], target_image.shape[1]
    image_bbox_dtls["dump_bbox"] = transform_bbox(boxes, target_img_size, pred_img_size)
    return image_bbox_dtls["dump_bbox"]


  def predict_stockpile(image_path):
    final_box = []
    crops = generate_crops(image_path, winW=1000, winH=1000, stepSize=500)
    img = cv2.imread(image_path)
    draw_temp = img.copy()

    for crop in tqdm(crops):
      img_ = img[crop[1]:crop[3], crop[0]:crop[2]]
      boxes = get_predictions(img_)

      if boxes:
        boxes = np.asarray(boxes)

        for box in boxes:
          b = (box.astype(int))
          if len(b) > 0:
            box = np.asarray(b)
            crop = np.asarray(crop)
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            x_final, y_final = b[0] + crop[0], b[1] + crop[1]
            final_box.append([x_final, y_final, x_final + (x2 - x1), y_final + (y2 - y1)])
    final_box = non_max_suppression(final_box)
    # boxarray =[]
    for box in final_box:
      draw_box(draw_temp, box, color=(0, 0, 255))
      print(box)
    cv2.imwrite(detectfn, draw_temp)

    # Saving the array in a text file
    np.savetxt(detecttxtfn, final_box.astype(int), delimiter=" ",fmt='%i')
    return final_box


    # intdata
    #infn = r"D:\Dropbox\P.Robeck\BPLA Dropbox\03 Planning\DN-Dumping Detection-1087\03_Data\GIS-data\satellite-WV3-042022\Dhahrat-Namar-World-view3-042022.png"

    infn = r"D:\Dropbox\P.Robeck\BPLA Dropbox\07 Temp\Drone-data\Dhibiyah\2_Aerial\GeoTiff\Full\Dhibiyah_Drone_GeoTiff_005.png"

    #detectfn = r"D:\Dropbox\P.Robeck\BPLA Dropbox\03 Planning\DN-Dumping Detection-1087\03_Data\GIS-data\satellite-WV3-042022\detect_Dhahrat-Namar-World-view3-042022.png"
    #detecttxtfn = r"D:\Dropbox\P.Robeck\BPLA Dropbox\03 Planning\DN-Dumping Detection-1087\03_Data\GIS-data\satellite-WV3-042022\detect_Dhahrat-Namar-World-view3-042022.txt"
    detectfn = r"D:\Dropbox\P.Robeck\BPLA Dropbox\07 Temp\Drone-data\Dhibiyah\2_Aerial\GeoTiff\Full\detected02_Dhibiyah_Drone_GeoTiff_005.png"
    detecttxtfn = r"D:\Dropbox\P.Robeck\BPLA Dropbox\07 Temp\Drone-data\Dhibiyah\2_Aerial\GeoTiff\Full\detected02_Dhibiyah_Drone_GeoTiff_005.txt"

    #detectfn = r'C:\Users\Robeck\PycharmProjects\tfPointnet\pointnet2\detected_22FEB15073634-S2AM-RCRC-4Bands-Mosaic-DN.png'
    # detecttxtfn = r'C:\\Users\\Robeck\\PycharmProjects\\tfPointnet\\pointnet2\\detected_22FEB15073634-S2AM-RCRC-4Bands-Mosaic-DN.txt'

    exists(infn)
    exists(detectfn)
    exists(detecttxtfn)

    predict_stockpile(infn)


if __name__ == "__main__":
    main()

def convert():
  #convert
  dataset = pd.read_csv(detecttxtfn, sep='\s+', header=None)
  outWld_path = r'C:\Users\Robeck\PycharmProjects\tfPointnet\pointnet2\22FEB15073634-S2AM-RCRC-4Bands-Mosaic-DN.wld'
  exists(outWld_path)

  # function to return polygon
  def bbox(x1, y1, x2, y2):
    # world file content
    wldcontent=[]
    with open(outWld_path) as f:
      while True:
        line = f.readline()
        if not line:
          break
        wldcontent.append(line.strip())

    # Line 1: A: x-component of the pixel width (x-scale)
    xscale = float(wldcontent[0])
    # Line 2: D: y-component of the pixel width (y-skew)
    yskew = float(wldcontent[1])
    # Line 3: B: x-component of the pixel height (x-skew)
    xskew = float(wldcontent[2])
    # Line 4: E: y-component of the pixel height (y-scale), typically negative
    yscale = float(wldcontent[3])
    # Line 5: C: x-coordinate of the center of the original image's upper left pixel transformed to the map
    xpos = float(wldcontent[4])
    # Line 6: F: y-coordinate of the center of the original image's upper left pixel transformed to the map
    ypos = float(wldcontent[5])

    ##print(long0, lat0, lat1, long1)

    X_proj = xpos + (xscale * x1) + (xskew * y1)
    Y_proj = ypos + (yscale * y1) + (yskew * x1)

    X1_proj = xpos + (xscale * x2) + (xskew * y2)
    Y1_proj = ypos + (yscale * y2) + (yskew * x2)

    return Polygon([[X_proj, Y_proj],
                    [X1_proj, Y_proj],
                    [X1_proj, Y1_proj],
                    [X_proj, Y1_proj]])

  outGDF = gpd.GeoDataFrame(geometry=dataset.apply(lambda g: bbox(int(g[0]), int(g[1]), int(g[2]), int(g[3])), axis=1),
                            crs={'init': 'epsg:32638'})
  # write to shape
  detectSHPfn = inPath+inProject+inFile+'/detected_'+inProject+'export_'+inFile+'.shp'

  exists(detectSHPfn)
  outGDF.to_file(detectSHPfn)


