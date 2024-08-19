TRAIN="
    all_six_datasets/ETT-small/ETTh1.csv 
    all_six_datasets/ETT-small/ETTh2.csv
    all_six_datasets/ETT-small/ETTm1.csv
    all_six_datasets/ETT-small/ETTm2.csv
    all_six_datasets/electricity/electricity.csv
    all_six_datasets/exchange_rate/exchange_rate.csv
    all_six_datasets/traffic/traffic.csv
    all_six_datasets/weather/weather.csv"


TEST="
    all_six_datasets/ETT-small/ETTh1.csv 
    all_six_datasets/ETT-small/ETTh2.csv
    all_six_datasets/ETT-small/ETTm1.csv
    all_six_datasets/ETT-small/ETTm2.csv
    all_six_datasets/electricity/electricity.csv
    all_six_datasets/exchange_rate/exchange_rate.csv
    all_six_datasets/traffic/traffic.csv
    all_six_datasets/weather/weather.csv"

PROMPT="prompt_bank/prompt_data_normalize_csv_split"
epoch=500
downsample_rate=20
freeze=0
lr=1e-3
model_path="/home/FINAL/gpt2-medium"


for pred_len in 96
do

    CUDA_VISIBLE_DEVICES=0 python3 main_ltsm.py \
    --model LTSM \
    --model_name_or_path ${model_path} \
    --train_epochs ${epoch} \
    --batch_size 800 \
    --pred_len ${pred_len} \
    --gradient_accumulation_steps 64 \
    --data_path ${TRAIN} \
    --test_data_path_list ${TEST} \
    --prompt_data_path ${PROMPT} \
    --freeze ${freeze} \
    --learning_rate ${lr} \
    --downsample_rate ${downsample_rate} \
    --output_dir "output/ltsm_csv_medium_lr${lr}_loraFalse_down${downsample_rate}_freeze${freeze}_e${epoch}_pred${pred_len}/"\
    --eval 1
done
