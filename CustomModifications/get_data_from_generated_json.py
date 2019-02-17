import numpy as np
import os, sys
import json
dependency = r"F:\Python\Segment2Data\CategoryDefination"
sys.path.append(dependency)
from CategoryDefination import TypeCategories
sys.path.remove(dependency)

class json_preprocessor(object):
  def __init__(self, json_dir_path):
    self.path_pfx = json_dir_path
    self.data = {}
    self.num_classes = len(TypeCategories) - 1
    self._preprocess_json()

  def _preprocess_json(self):
    filenames = os.listdir(self.path_pfx)
    for filename in filenames:
      json_root = json.loads(open(os.path.join(self.path_pfx, filename), "r").read())
      width = json_root["imageInformation"]["imageWidth"]
      height = json_root["imageInformation"]["imageHeight"]
      bboxes = []
      one_hot_classes = []
      for bbox_key in json_root["annotations"]:
        annotation = json_root["annotations"][bbox_key]
        xmin = annotation["xmin"] / width
        xmax = annotation["xmax"] / width
        ymin = annotation["ymin"] / height
        ymax = annotation["ymax"] / height
        bbox = [xmin,ymin,xmax,ymax]
        bboxes.append(bbox)
        one_hot_class = self._to_one_hot_array(TypeCategories[annotation["Category"]].value)
        one_hot_classes.append(one_hot_class)
      image_name = json_root["imageInformation"]["imageName"]
      bboxes = np.asarray(bboxes)
      one_hot_classes = np.asarray(one_hot_classes)
      image_data = np.hstack((bboxes, one_hot_classes))
      self.data[image_name] = image_data

  def _to_one_hot_array(self, idx):
    arr = [0] * self.num_classes
    arr[idx] = 1
    return arr

def _main():
  #print(json_preprocessor("F:\Python\Segment2Data\Annotations").data)
  import pickle
  data = json_preprocessor("F:\Python\Segment2Data\Annotations").data
  pickle.dump(data, open('UnityGen.p', 'wb'))

if __name__ == "__main__":
  _main()