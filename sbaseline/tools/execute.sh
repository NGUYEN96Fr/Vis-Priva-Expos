#!/bin/bash

python3 main.py --config_file ../configs/mobi_BL.yaml --model_name bank_mobi.pkl --situation BANK
python3 main.py --config_file ../configs/mobi_BL.yaml --model_name it_mobi.pkl --situation IT
python3 main.py --config_file ../configs/mobi_BL.yaml --model_name wait_mobi.pkl --situation WAIT
python3 main.py --config_file ../configs/mobi_BL.yaml --model_name accom_mobi.pkl --situation ACCOM
python3 main.py --config_file ../configs/rcnn_BL.yaml --model_name bank_rcnn.pkl --situation BANK
python3 main.py --config_file ../configs/rcnn_BL.yaml --model_name it_rcnn.pkl --situation IT
python3 main.py --config_file ../configs/rcnn_BL.yaml --model_name wait_rcnn.pkl --situation WAIT
python3 main.py --config_file ../configs/rcnn_BL.yaml --model_name accom_rcnn.pkl --situation ACCOM