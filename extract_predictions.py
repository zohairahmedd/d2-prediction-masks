from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import argparse
import cv2
import numpy as np

def main ():
    """
    take command line arguments input for the paths of the location of the image

    arguments:
        none
    """
    parser = argparse.ArgumentParser(description='processing some image') # necessary for implementing command-line
    parser.add_argument('input_image_path', type=str, help='path to image') # adds argument input_image_path to the parser

    args = parser.parse_args() # allows us to use the arguments in the parser (args.argument_name)

    image_to_annotate(args.input_image_path)

def image_to_annotate(input_image_path):
    """
    annotate any file using detectron2 default prediction model

    arguments:
        input_image_path (str): path to image file
    """
    # configuration file for the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda"

    # initialize the predictor
    predictor = DefaultPredictor(cfg)

    # load file
    img = cv2.imread(input_image_path)

    # convert the image from BGR to RGB format expected by detectron2
    img_rgb = img[:, :, ::-1]

    # run detection
    outputs = predictor(img_rgb)

    # visualise
    visualized = Visualizer(img_rgb, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    visualized = visualized.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualized = visualized.get_image()[:, :, ::-1]

    # metadata for the dataset
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # checks if instance has been detected
    if len(outputs["instances"].pred_boxes.tensor) > 0:
    # extract the predicted masks and classes
        pred_masks = outputs["instances"].pred_masks.cpu().numpy()
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()

    # iterate over found masks and calculate their areas
        for mask, cls in zip(pred_masks, pred_classes):
            area = np.sum(mask)  # calculate the area by summing all true values (sum of all pixels set to 1, assuming binary mask)
            label = metadata.thing_classes[int(cls)] # meaningful display/storage of detected object types
            print(f"Label: {label}, Mask area: {area}")

    else:
        print("No more instances detected.")

    cv2.imshow("visualized", visualized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()