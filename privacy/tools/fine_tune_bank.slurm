#!/bin/bash
# SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition allcpu
#SBATCH -w node01
#SBATCH -J FT
# SBATCH --output=fine_tune_bank.out
# SBATCH --error=fine_tune_bank.errs

python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_0.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_1.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_2.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_3.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_4.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_5.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_6.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_7.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_8.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_9.pkl --situation BANK --opts USER_SELECTOR.EPS 0.05 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_10.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_11.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_12.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_13.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_14.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_15.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_16.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_17.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_18.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_19.pkl --situation BANK --opts USER_SELECTOR.EPS 0.1 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_20.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_21.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_22.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_23.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_24.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_25.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_26.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_27.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_28.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_29.pkl --situation BANK --opts USER_SELECTOR.EPS 0.15 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_30.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_31.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.8 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_32.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_33.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.85 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_34.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_35.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.9 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_36.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_37.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 0.95 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_38.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE VOTE OUTPUT.DIR fine_tune_bank &
python3 ft_official.py --config_file ../configs/rf_kmeans_ft_mobi_cv5.yaml --model_name fine_tune_bank_39.pkl --situation BANK --opts USER_SELECTOR.EPS 0.2 USER_SELECTOR.KEEP 1.0 SOLVER.FEATURE_TYPE ORG OUTPUT.DIR fine_tune_bank &
wait
