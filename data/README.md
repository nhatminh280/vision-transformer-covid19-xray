# Data Directory

This directory contains the COVID-19 Radiography Database and processed datasets.

## Structure

```
data/
├── raw/                           # Original dataset
│   └── COVID-19_Radiography_Dataset/
│       ├── COVID/                 # COVID-19 cases
│       ├── Normal/                # Normal cases
│       ├── Lung_Opacity/          # Lung opacity cases
│       └── Viral_Pneumonia/       # Viral pneumonia cases
├── processed/                     # Processed dataset
│   ├── train/                     # Training set (70%)
│   ├── val/                       # Validation set (15%)
│   └── test/                      # Test set (15%)
└── metadata/                      # Metadata files
    ├── train_metadata.csv
    ├── val_metadata.csv
    └── test_metadata.csv
```

## Dataset Information

- **Source**: COVID-19 Radiography Database
- **Total Images**: ~21,000 images
- **Classes**: 4 classes
  - COVID-19: ~3,600 images
  - Normal: ~10,200 images
  - Lung Opacity: ~6,000 images
  - Viral Pneumonia: ~1,300 images
- **Format**: PNG images
- **Resolution**: 299×299 pixels

## Usage

1. Download the dataset using `scripts/download_dataset.py`
2. Process the data using `scripts/prepare_data.py`
3. The processed data will be automatically organized into train/val/test splits

## Data Preprocessing

- Images are resized to 224×224 pixels
- Normalized using ImageNet statistics
- Data augmentation applied during training
- Balanced sampling for handling class imbalance

**\***COVID-19 CHEST X-RAY DATABASE

A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal and other lung infection dataset is released in stages. In the first release we have released 219 COVID-19, 1341 normal and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection) and 1345 Viral Pneumonia images and corresponding lung masks. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.

## \*\*COVID-19 data:

COVID data are collected from different publicly accessible dataset, online sources and published papers.
-2473 CXR images are collected from padchest dataset[1].
-183 CXR images from a Germany medical school[2].
-559 CXR image from SIRM, Github, Kaggle & Tweeter[3,4,5,6]
-400 CXR images from another Github source[7].

\*\*\*Normal images:

---

10192 Normal data are collected from from three different dataset.
-8851 RSNA [8]
-1341 Kaggle [9]

\*\*\*Lung opacity images:

---

6012 Lung opacity CXR images are collected from Radiological Society of North America (RSNA) CXR dataset [8]

\*\*\*Viral Pneumonia images:

---

1345 Viral Pneumonia data are collected from the Chest X-Ray Images (pneumonia) database [9]

Please cite the follwoing two articles if you are using this dataset:
-M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
-Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.

\*\*Reference:
[1]https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711
[2]https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png
[3]https://sirm.org/category/senza-categoria/covid-19/
[4]https://eurorad.org
[5]https://github.com/ieee8023/covid-chestxray-dataset
[6]https://figshare.com/articles/COVID-19_Chest_X-Ray_Image_Repository/12580328
[7]https://github.com/armiro/COVID-CXNet  
[8]https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
[9] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

*\*\*Formats - All the images are in Portable Network Graphics (PNG) file format and resolution are 299*299 pixels.

\*\*\*\*Objective - Researchers can use this database to produce useful and impactful scholarly work on COVID-19, which can help in tackling this pandemic.

**\***COVID-19 CHEST X-RAY DATABASE

A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal and other lung infection dataset is released in stages. In the first release we have released 219 COVID-19, 1341 normal and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection) and 1345 Viral Pneumonia images and corresponding lung masks. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.

## \*\*COVID-19 data:

COVID data are collected from different publicly accessible dataset, online sources and published papers.
-2473 CXR images are collected from padchest dataset[1].
-183 CXR images from a Germany medical school[2].
-559 CXR image from SIRM, Github, Kaggle & Tweeter[3,4,5,6]
-400 CXR images from another Github source[7].

\*\*\*Normal images:

---

10192 Normal data are collected from from three different dataset.
-8851 RSNA [8]
-1341 Kaggle [9]

\*\*\*Lung opacity images:

---

6012 Lung opacity CXR images are collected from Radiological Society of North America (RSNA) CXR dataset [8]

\*\*\*Viral Pneumonia images:

---

1345 Viral Pneumonia data are collected from the Chest X-Ray Images (pneumonia) database [9]

Please cite the follwoing two articles if you are using this dataset:
-M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
-Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.

\*\*Reference:
[1]https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711
[2]https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png
[3]https://sirm.org/category/senza-categoria/covid-19/
[4]https://eurorad.org
[5]https://github.com/ieee8023/covid-chestxray-dataset
[6]https://figshare.com/articles/COVID-19_Chest_X-Ray_Image_Repository/12580328
[7]https://github.com/armiro/COVID-CXNet  
[8]https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
[9] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

*\*\*Formats - All the images are in Portable Network Graphics (PNG) file format and resolution are 299*299 pixels.

\*\*\*\*Objective - Researchers can use this database to produce useful and impactful scholarly work on COVID-19, which can help in tackling this pandemic.
