## Extracting the Detectron2 Predicted Masks
Developed a script using Python that will predict annotations using the Detectron2 model and extract the masks
- Utilized Detectron2, OpenCV, & NumPy libraries

## Commands
- create a virtual environment
- download pytorch: python3 -m pip install torch torchvision, python3 -m pip install Pillow==9.5.0, python3 -m pip install numpy==1.23.1
- download detectron2: python3 -m pip install detectron2 OR python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
- download opencv: python3 -m pip install opencv-python
- download numpy: python3 -m pip install numpy
- run with "python3 extract_predictions.py (path to image) (path for the directory to store the annotated image)"

## Example of Running Script (absolute path)
- python3 extract_predictions.py C:\\\directory\\\cat.jpg C:directory\\\annotated_images
- annotated_images will be created if it does not exist at specified location

## Extra
- included test image as an example