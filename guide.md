### Step 1: Install requirements 
```bash 
pip install -r requirements.txt
```

### Step 2: Use this line to download data 
```bash
wget 'https://cos.twcc.ai/fishlentrafficdataset/Fisheye8K_all_including_train%26test_update_2024Jan.zip'
```

```bash
unzip "Fisheye8K_all_including_train&test_update_2024Jan.zip" -d Fisheye8K && rm "Fisheye8K_all_including_train&test_update_2024Jan.zip"
```

### Step 3: Run restruct data 
```bash 
python restruct_data.py
```

### Step 4: Run split data 
```bash 
python split_data.py
```

### Step 5: Start training 
```bash
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --eval
```
or 
```bash 
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --fleet --eval
```

### Step 6: Evaluation
```bash 
python tools/eval.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv3_r18vd_6x_coco.pdparams
```