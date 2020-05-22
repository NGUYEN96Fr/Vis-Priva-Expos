# Vis-Priva-Expos
Visual Privacy Exposure

## Prequisites
You should have the **annotations** and **inferences** directories in the Vis-Priva-Expos.

## Running the tests
There're two steps to run user visual privacy  exposure evaluation :

**1. Running the objectness detection threshold determination file:**
```
python .\dectection_threshold.py .\param_config.ini
```
Currently, it supports two methods, which are the thresholding method based on an indiviual category characteristic and the other based on a common FDR for all categories. To run it, set up the configuration file as following:
- Indiviual objectness threshold:
```
[dectection_threshold.py]
##############INPUT##############
#ground truth bounding box path file
gt_path = ./inferences/verite_terrain.txt

#predicted bounding box path file
prd_path= ./inferences/test_set.txt

#IoU threshold
iou_thresh = 0.5

#load preprocessed bounding box files.
load_bb = False

#load selected predictions(IoUs > iou_thresh), aldready done before
load_sel_prd = False

#Find optimal indiviual objectness threshold
indi_thresh_flag = True

#save name threshold detection file
sav_name = objness_thresh_20.txt

#False Discovery Rates
FDRs = 0.20

```
It will not use the **FDRs** defined in the configuaration file, but It will use a range of FDRs from 0 to 0.2 predefined in the threshold detection codes. If you have aldready preprocessed the **verite_terrain.txt** and **test_set.txt** at the first time of the execution, at second time you just need to load them by setting **load_bb = True**, **load_sel_prd = True** to accelarate the execution. 

The result will be the **individual_threshold.txt** file, which showes an optimal objectness threshold for each category.

- A Common FDR for all categories
```
[dectection_threshold.py]
##############INPUT##############
#ground truth bounding box path file
gt_path = ./inferences/verite_terrain.txt

#predicted bounding box path file
prd_path= ./inferences/test_set.txt

#IoU threshold
iou_thresh = 0.5

#load preprocessed bounding box files.
load_bb = False

#load selected predictions(IoUs > iou_thresh), aldready done before
load_sel_prd = False


#Find optimal indiviual objectness threshold
indi_thresh_flag = False

#save name threshold detection file
sav_name = objness_thresh_20.txt

#False Discovery Rates
FDRs = 0.20
```
You only need to set **indi_thresh_flag = False**, the code will use the **FDRs** in the configuration file, and save a result in the **sav_name**.

**2. Running the user visual privacy exoposure evaluation:**
```
python .\user_exposure.py .\param_config.ini
```
The user exposure evaluation depends on the used objectness threshold methods.Currently it supports the two methods as above.
- The threholding method based on an individual category.
```
[user_exposure.py]
##############INPUT##############
#situation files
situ_path = ./annotations/

#user inference files
user_path = ./inferences/gt_users_inferences.txt

#objectness threshold file
obj_path = individual_threshold.txt

#save path
sav_path = ./users_ranking/

#objectness threshold method in one of [fdr_indiv , fdr_class_x]
method = fdr_indiv
```
You only need to set up **method = fdr_indiv** and choose the corresponding objectness threshold file **obj_path = individual_threshold.txt = individual_threshold.txt**. The result will be saved in **sav_path**.

- The thresholding method based on a common FDR for all categories
```
[user_exposure.py]
##############INPUT##############
#situation files
situ_path = ./annotations/

#user inference files
user_path = ./inferences/gt_users_inferences.txt

#objectness threshold file
obj_path = objness_thresh_20.txt

#save path
sav_path = ./users_ranking/

#objectness threshold method in one of [fdr_indiv , fdr_class_x]
method = fdr_class_20
```
Set up the **method = fdr_class_x** where x is the used current FDR (e.g. fdr_class_20 means the fdr = 0.2) and choose the corresponding **obj_path** file. The result will be saved in the **sav_path** file.
