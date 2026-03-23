# 100-Class Animal Bounding Box Annotation Dataset (A100-Det)



# 1. Dataset Overview

## 1.1 Background and Significance of the Dataset

Animal object detection is a crucial research direction in computer vision, widely applied in wildlife conservation, intelligent surveillance, agricultural pest control, biodiversity surveys, and other domains. High-quality, large-scale annotated datasets serve as the core foundation for enhancing the performance of animal object detection models. However, current mainstream authoritative datasets still have certain limitations:

- **COCO Dataset**: As a benchmark dataset in object detection, it includes some animal categories but covers a limited range of species (only dozens of species). Additionally, the sample size of individual animal categories is unbalanced, and samples of some rare or niche animals are missing, making it difficult to meet refined training requirements in specific scenarios.

- **ImageNet Dataset**: Focused on image classification tasks, it boasts a rich variety of animal species but lacks bounding box annotation information for object detection, preventing direct use in the training and validation of detection models.

- **PASCAL VOC Dataset**: It only includes a few common animal species (e.g., cats, dogs, and birds), with extremely low species coverage and a small sample size, which fails to support model training in complex scenarios.

To address these issues, this dataset (A100-Det) is constructed as a bounding box annotation dataset containing 100 animal species, with the sample size per species controlled between 100 and 130 images. This design achieves extensive coverage of animal species and sufficient supply of single-category samples. As of now, no public dataset can simultaneously meet the core requirements of "over 100 animal species, single-category sample size ≥ 100 images, and high-precision bounding box annotations". This dataset effectively fills this research gap and provides high-quality training and testing data support for the development and optimization of animal object detection models, particularly advancing research in subfields such as niche animal detection and multi-animal mixed scene detection.



## 1.2 Basic Information of the Dataset

- Number of animal species: 100 (see class.txt for specific species, sorted alphabetically by the first letter of their names)

- Total number of samples: 11,354 images

- Single-category sample size: 100-130 images per species

- Annotation type: Bounding box (object detection)

- Annotation formats: X-AnyLabeling original JSON format, VOC XML format

- Image format: PNG

The number of images for each animal species in the dataset is shown in the figure below:

![](./pic/count.png)



# 2. Data Acquisition and Annotation Process

## 2.1 Image Acquisition

All images in this dataset are obtained from Bing Image Search (https://cn.bing.com/images) in strict compliance with copyright regulations. The specific process is as follows:

1. Access the Bing Image Search page and use the filter to select the "Free to modify, share, and use" permission to ensure compliant use of images;

2. Take antelope as an example; the search link is: https://cn.bing.com/images/search?q=antelope&qs=n&form=QBIR&qft=%20filterui%3Alicense-L2_L3_L5_L6. Repeat this process to search for images of all 100 animal species;

3. Use the Microsoft Edge browser extension ImageAssistant to batch extract and download images from the current search page;

4. Manual cleaning: Remove blurred images, those with severe occlusion, extremely low resolution, unclear animal subjects, or potential copyright disputes to ensure the usability of each image.



## 2.2 Annotation Process

We adopt a semi-automatic annotation approach combining AI assistance and manual correction to balance annotation efficiency and accuracy. The specific process is as follows:

1. Annotation tool: Utilize the open-source annotation tool X-AnyLabeling (https://github.com/CVHub520/X-AnyLabeling);

2. AI-assisted annotation: Employ the YOLOv8m model, set the confidence threshold to 0.4 and the Intersection over Union (IoU) threshold to 0.8, and conduct batch pre-annotation on all cleaned images;

3. Manual correction: Review AI pre-annotation results one by one, manually correct bounding box position deviations, missing annotations, and misannotations; manually annotate animal targets not detected by AI to ensure accurate annotation of the animal subject in each image;

4. Format export: After annotation, retain the original JSON annotation files from X-AnyLabeling and export them as VOC XML format files to meet the training needs of mainstream object detection models (e.g., YOLO series, Faster R-CNN).



# 3. Dataset Structure

The dataset adopts a clear hierarchical folder structure to facilitate quick data location and usage. The overall structure is as follows:

```makefile
animal/                  # Main folder
├─ images/               # Image folder
│  ├─ antelope/          # Animal-specific subfolders (sorted by class.txt)
│  │  ├─ antelope_001.png
│  │  ├─ antelope_002.png
│  │  └─ ... (100-130 images in total)
│  ├─ bear/
│  │  └─ ...
│  └─ ... (100 animal subfolders in total)
├─ annotations/          # Annotation folder
│  ├─ raw_json/          # X-AnyLabeling original JSON annotation files
│  │  ├─ antelope/
│  │  │  ├─ antelope_001.json
│  │  │  └─ ...
│  │  └─ ... (100 animal subfolders in total)
│  └─ voc_xml/           # VOC format XML annotation files
│     ├─ antelope/
│     │  ├─ antelope_001.xml
│     │  └─ ...
│     └─ ... (100 animal subfolders in total)
└─ class.txt             # List of animal species (sorted alphabetically by first letter, one species per line)
```

Notes:

- class.txt: Strictly sorted alphabetically by the first letter of animal names, serving as the naming reference for subfolders in both the images and annotations folders to ensure consistent data correspondence;

- Image naming rule: "animal_name_xxx.png", where "xxx" is a three-digit number (001-130), facilitating batch reading and sample management;

- Annotation file naming: Exactly the same as the corresponding image name to ensure one-to-one correspondence between images and annotations.



# 4. Examples of Annotation Files

## 4.1 X-AnyLabeling Original JSON Format (taking antelope_001.json as an example)

```json
{
  "version": "3.3.9",          // X-AnyLabeling tool version number
  "flags": {},                 // Custom tags (not used in this dataset)
  "shapes": [                  // List of bounding boxes (only one animal subject is annotated per image)
    {
      "label": "antelope",     // Annotation category (animal name)
      "score": 0.899715781211853,  // Confidence of AI pre-annotation (when AI annotation is inaccurate, only the position of the bounding box is manually modified, and the confidence can be ignored)
      "points": [              // Coordinates of the four vertices of the bounding box (clockwise order: top-left, top-right, bottom-right, bottom-left)
        [371.7910432815552, 121.02197647094727],
        [947.3024787902832, 121.02197647094727],
        [947.3024787902832, 972.2155342102051],
        [371.7910432815552, 972.2155342102051]
      ],
      "group_id": null,        // Annotation group ID (not used in this dataset)
      "description": null,     // Annotation description (not used in this dataset)
      "difficult": false,      // Whether it is a difficult sample (all samples in this dataset are easy to annotate)
      "shape_type": "rectangle",  // Annotation shape (bounding box)
      "flags": {},             // Custom tags for bounding boxes (not used in this dataset)
      "attributes": {},        // Annotation attributes (not used in this dataset)
      "kie_linking": []        // KIE linking information (not used in this dataset)
    }
  ],
  "imagePath": "antelope_001.png",  // Corresponding image path
  "imageData": null,           // Image Base64 encoding (not stored in this dataset to save space)
  "imageHeight": 1025,         // Image height (pixels)
  "imageWidth": 1640,          // Image width (pixels)
  "description": ""            // Image description (not used in this dataset)
}
```



## 4.2 VOC XML Format (antelope_001.xml)

```xml
<?xml version="1.0" ?&gt;
&lt;annotation&gt;                 <!-- Annotation root node -->
  <folder>H:/project/Animal_position/animal/annotations</folder&gt;  <!-- Path of the folder where the annotation file is located -->
  &lt;filename&gt;antelope_001.png&lt;/filename&gt;  <!-- Corresponding image file name -->
  &lt;size&gt;<!-- Image size information -->
    <width>1640&lt;/width&gt;       <!-- Image width (pixels) -->
    <height>1025&lt;/height&gt;     <!-- Image height (pixels) -->
    <depth>3&lt;/depth&gt;          <!-- Number of image channels (3 for RGB) -->
  &lt;/size&gt;
  &lt;source&gt;                   <!-- Data source -->
    <database>https://github.com/CVHub520/X-AnyLabeling</database&gt;  <!-- Source of the annotation tool -->
  </source>
  <object&gt;<!-- Target object information -->
    <name>antelope</name&gt;     <!-- Target category (animal name) -->
    <pose&gt;Unspecified&lt;/pose&gt;  <!-- Target pose (unspecified) -->
    &lt;truncated&gt;0&lt;/truncated&gt;  <!-- Whether truncated (0 = not truncated, 1 = truncated) -->
    <occluded>0</occluded&gt;    <!-- Whether occluded (0 = no occlusion, 1 = occluded) -->
    &lt;difficult&gt;0&lt;/difficult&gt;  <!-- Whether it is a difficult sample (0 = easy to annotate, 1 = difficult) -->
    <bndbox>                 <!-- Bounding box coordinates (VOC format, top-left corner is (0,0)) -->
      &lt;xmin&gt;371&lt;/xmin&gt;        <!-- Top-left x coordinate -->
      &lt;ymin&gt;121&lt;/ymin&gt;        <!-- Top-left y coordinate -->
      &lt;xmax&gt;947&lt;/xmax&gt;        <!-- Bottom-right x coordinate -->
      &lt;ymax&gt;972&lt;/ymax&gt;        <!-- Bottom-right y coordinate -->
    </bndbox>
  </object>
</annotation>
```



# 5. Model Training Results

This dataset has been initially trained on YOLO series models, and the following table presents the model performance metrics:

![](./pic/table1.png)

![](./pic/table2.png)



Here, YOLOv8m is taken as an example to demonstrate some training output images. For more details, refer to the Kaggle links provided below:

![](./pic/v8m.png)

![](./pic/confusion.png)

![](./pic/val.jpg)



# 6. Dataset Usage Instructions

- This dataset can be directly used for the training, validation, and testing of object detection models, supporting mainstream models such as the YOLO series, Faster R-CNN, and SSD;

- Note: Files in the images folder correspond one-to-one with those in the annotations folder. class.txt serves as the basis for category configuration during model training, and the category order must not be altered;

- VOC XML format annotation files can be directly used for model training. If conversion to other formats (e.g., COCO JSON) is required, the X-AnyLabeling tool or third-party scripts can be utilized;

Kaggle training log links:

- [A100-Det_YOLOv8n](https://www.kaggle.com/code/rexinshiminxiaozhou/animal-position-box-yolov8n)

- [A100-Det_YOLOv8s](https://www.kaggle.com/code/rexinshiminxiaozhou/animal-position-box-yolov8s)

- [A100-Det_YOLOv8m](https://www.kaggle.com/code/rexinshiminxiaozhou/animal-position-box-yolov8m)

- [A100-Det_YOLOv10s](https://www.kaggle.com/code/rexinshiminxiaozhou/animal-position-box-yolov10s)

- [AnDD_YOLOv8n,s (i.e., Animal Detection Dataset)](https://www.kaggle.com/code/rexinshiminxiaozhou/add-animal-yolov8n-s)

- [WiDD_YOLOv8n,s (i.e., Wildlife Detection Dataset)](https://www.kaggle.com/code/rexinshiminxiaozhou/wdd-animal-yolov8n-s)

- [COCO_YOLOv8n,s (i.e., Common Objects in Context)](https://www.kaggle.com/code/rexinshiminxiaozhou/coco-animal-yolov8n-s)
