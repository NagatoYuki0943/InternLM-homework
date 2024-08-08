# OpenCompass 评测 InternLM-1.8B 实践

1. 安装 opencompass

```sh
mkdir opencompass
cd opencompass
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -v -e .
```

![image-20240808143453633](1-7_opencompass.assets/image-20240808143453633.png)

2. link 模型

```sh
(langgpt) root@intern-studio-030876:~/opencompass# mkdir models
(langgpt) root@intern-studio-030876:~/opencompass# cd models
(langgpt) root@intern-studio-030876:~/opencompass/models# ln -s /share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat/ ./
(langgpt) root@intern-studio-030876:~/opencompass/models# ls
internlm2_5-1_8b-chat
```

3. 数据准备

解压评测数据集到 data/ 处

```
cd /root/opencompass/opencompass
cp /share/temp/datasets/OpenCompassData-core-20231110.zip ./
unzip OpenCompassData-core-20231110.zip
```

![image-20240808143914826](1-7_opencompass.assets/image-20240808143914826.png)

4.查看支持的数据集和模型

列出所有跟 InternLM 及 C-Eval 相关的配置

```
(langgpt) root@intern-studio-030876:~/opencompass/opencompass# python tools/list_configs.py internlm ceval
+----------------------------------------+----------------------------------------------------------------------+
| Model                                  | Config Path                                                          |
|----------------------------------------+----------------------------------------------------------------------|
| hf_internlm2_1_8b                      | configs/models/hf_internlm/hf_internlm2_1_8b.py                      |
| hf_internlm2_20b                       | configs/models/hf_internlm/hf_internlm2_20b.py                       |
| hf_internlm2_5_1_8b_chat               | configs/models/hf_internlm/hf_internlm2_5_1_8b_chat.py               |
| hf_internlm2_5_20b_chat                | configs/models/hf_internlm/hf_internlm2_5_20b_chat.py                |
| hf_internlm2_5_7b                      | configs/models/hf_internlm/hf_internlm2_5_7b.py                      |
| hf_internlm2_5_7b_chat                 | configs/models/hf_internlm/hf_internlm2_5_7b_chat.py                 |
| hf_internlm2_7b                        | configs/models/hf_internlm/hf_internlm2_7b.py                        |
| hf_internlm2_base_20b                  | configs/models/hf_internlm/hf_internlm2_base_20b.py                  |
| hf_internlm2_base_7b                   | configs/models/hf_internlm/hf_internlm2_base_7b.py                   |
| hf_internlm2_chat_1_8b                 | configs/models/hf_internlm/hf_internlm2_chat_1_8b.py                 |
| hf_internlm2_chat_1_8b_sft             | configs/models/hf_internlm/hf_internlm2_chat_1_8b_sft.py             |
| hf_internlm2_chat_20b                  | configs/models/hf_internlm/hf_internlm2_chat_20b.py                  |
| hf_internlm2_chat_20b_sft              | configs/models/hf_internlm/hf_internlm2_chat_20b_sft.py              |
| hf_internlm2_chat_20b_with_system      | configs/models/hf_internlm/hf_internlm2_chat_20b_with_system.py      |
| hf_internlm2_chat_7b                   | configs/models/hf_internlm/hf_internlm2_chat_7b.py                   |
| hf_internlm2_chat_7b_sft               | configs/models/hf_internlm/hf_internlm2_chat_7b_sft.py               |
| hf_internlm2_chat_7b_with_system       | configs/models/hf_internlm/hf_internlm2_chat_7b_with_system.py       |
| hf_internlm2_chat_math_20b             | configs/models/hf_internlm/hf_internlm2_chat_math_20b.py             |
| hf_internlm2_chat_math_20b_with_system | configs/models/hf_internlm/hf_internlm2_chat_math_20b_with_system.py |
| hf_internlm2_chat_math_7b              | configs/models/hf_internlm/hf_internlm2_chat_math_7b.py              |
| hf_internlm2_chat_math_7b_with_system  | configs/models/hf_internlm/hf_internlm2_chat_math_7b_with_system.py  |
| hf_internlm2_math_20b                  | configs/models/hf_internlm/hf_internlm2_math_20b.py                  |
| hf_internlm2_math_7b                   | configs/models/hf_internlm/hf_internlm2_math_7b.py                   |
| hf_internlm_20b                        | configs/models/hf_internlm/hf_internlm_20b.py                        |
| hf_internlm_7b                         | configs/models/hf_internlm/hf_internlm_7b.py                         |
| hf_internlm_chat_20b                   | configs/models/hf_internlm/hf_internlm_chat_20b.py                   |
| hf_internlm_chat_7b                    | configs/models/hf_internlm/hf_internlm_chat_7b.py                    |
| internlm_7b                            | configs/models/internlm/internlm_7b.py                               |
| lmdeploy_internlm2_1_8b                | configs/models/hf_internlm/lmdeploy_internlm2_1_8b.py                |
| lmdeploy_internlm2_20b                 | configs/models/hf_internlm/lmdeploy_internlm2_20b.py                 |
| lmdeploy_internlm2_5_1_8b_chat         | configs/models/hf_internlm/lmdeploy_internlm2_5_1_8b_chat.py         |
| lmdeploy_internlm2_5_20b_chat          | configs/models/hf_internlm/lmdeploy_internlm2_5_20b_chat.py          |
| lmdeploy_internlm2_5_7b                | configs/models/hf_internlm/lmdeploy_internlm2_5_7b.py                |
| lmdeploy_internlm2_5_7b_chat           | configs/models/hf_internlm/lmdeploy_internlm2_5_7b_chat.py           |
| lmdeploy_internlm2_5_7b_chat_1m        | configs/models/hf_internlm/lmdeploy_internlm2_5_7b_chat_1m.py        |
| lmdeploy_internlm2_7b                  | configs/models/hf_internlm/lmdeploy_internlm2_7b.py                  |
| lmdeploy_internlm2_base_20b            | configs/models/hf_internlm/lmdeploy_internlm2_base_20b.py            |
| lmdeploy_internlm2_base_7b             | configs/models/hf_internlm/lmdeploy_internlm2_base_7b.py             |
| lmdeploy_internlm2_chat_1_8b           | configs/models/hf_internlm/lmdeploy_internlm2_chat_1_8b.py           |
| lmdeploy_internlm2_chat_1_8b_sft       | configs/models/hf_internlm/lmdeploy_internlm2_chat_1_8b_sft.py       |
| lmdeploy_internlm2_chat_20b            | configs/models/hf_internlm/lmdeploy_internlm2_chat_20b.py            |
| lmdeploy_internlm2_chat_20b_sft        | configs/models/hf_internlm/lmdeploy_internlm2_chat_20b_sft.py        |
| lmdeploy_internlm2_chat_7b             | configs/models/hf_internlm/lmdeploy_internlm2_chat_7b.py             |
| lmdeploy_internlm2_chat_7b_sft         | configs/models/hf_internlm/lmdeploy_internlm2_chat_7b_sft.py         |
| lmdeploy_internlm2_series              | configs/models/hf_internlm/lmdeploy_internlm2_series.py              |
| lmdeploy_internlm_20b                  | configs/models/hf_internlm/lmdeploy_internlm_20b.py                  |
| lmdeploy_internlm_7b                   | configs/models/hf_internlm/lmdeploy_internlm_7b.py                   |
| lmdeploy_internlm_chat_20b             | configs/models/hf_internlm/lmdeploy_internlm_chat_20b.py             |
| lmdeploy_internlm_chat_7b              | configs/models/hf_internlm/lmdeploy_internlm_chat_7b.py              |
| ms_internlm_chat_7b_8k                 | configs/models/ms_internlm/ms_internlm_chat_7b_8k.py                 |
| vllm_internlm2_chat_1_8b               | configs/models/hf_internlm/vllm_internlm2_chat_1_8b.py               |
| vllm_internlm2_chat_1_8b_sft           | configs/models/hf_internlm/vllm_internlm2_chat_1_8b_sft.py           |
| vllm_internlm2_chat_20b                | configs/models/hf_internlm/vllm_internlm2_chat_20b.py                |
| vllm_internlm2_chat_20b_sft            | configs/models/hf_internlm/vllm_internlm2_chat_20b_sft.py            |
| vllm_internlm2_chat_7b                 | configs/models/hf_internlm/vllm_internlm2_chat_7b.py                 |
| vllm_internlm2_chat_7b_sft             | configs/models/hf_internlm/vllm_internlm2_chat_7b_sft.py             |
| vllm_internlm2_series                  | configs/models/hf_internlm/vllm_internlm2_series.py                  |
+----------------------------------------+----------------------------------------------------------------------+
+--------------------------------+------------------------------------------------------------------+
| Dataset                        | Config Path                                                      |
|--------------------------------+------------------------------------------------------------------|
| ceval_clean_ppl                | configs/datasets/ceval/ceval_clean_ppl.py                        |
| ceval_contamination_ppl_810ec6 | configs/datasets/contamination/ceval_contamination_ppl_810ec6.py |
| ceval_gen                      | configs/datasets/ceval/ceval_gen.py                              |
| ceval_gen_2daf24               | configs/datasets/ceval/ceval_gen_2daf24.py                       |
| ceval_gen_5f30c7               | configs/datasets/ceval/ceval_gen_5f30c7.py                       |
| ceval_internal_ppl_1cd8bf      | configs/datasets/ceval/ceval_internal_ppl_1cd8bf.py              |
| ceval_internal_ppl_93e5ce      | configs/datasets/ceval/ceval_internal_ppl_93e5ce.py              |
| ceval_ppl                      | configs/datasets/ceval/ceval_ppl.py                              |
| ceval_ppl_1cd8bf               | configs/datasets/ceval/ceval_ppl_1cd8bf.py                       |
| ceval_ppl_578f8d               | configs/datasets/ceval/ceval_ppl_578f8d.py                       |
| ceval_ppl_93e5ce               | configs/datasets/ceval/ceval_ppl_93e5ce.py                       |
| ceval_zero_shot_gen_bd40ef     | configs/datasets/ceval/ceval_zero_shot_gen_bd40ef.py             |
+--------------------------------+------------------------------------------------------------------+
```

5. 启动评测

```sh
export MKL_SERVICE_FORCE_INTEL=1
# 或
export MKL_THREADING_LAYER=GNU

python run.py \
--datasets ceval_gen \
--hf-path ../models/internlm2_5-1_8b-chat \
--hf-type chat \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--model-kwargs trust_remote_code=True device_map='auto' \
--generation-kwargs do_sample=True top_k=50 top_p=0.95 \
--max-seq-len 1024 \
--max-out-len 16 \
--batch-size 2 \
--hf-num-gpus 1 \
--debug
```

![image-20240808145034712](1-7_opencompass.assets/image-20240808145034712.png)

评测结果如下

```sh
dataset                                         version    metric         mode      internlm2_5-1_8b-chat_hf
----------------------------------------------  ---------  -------------  ------  --------------------------
ceval-computer_network                          db9ce2     accuracy       gen                          47.37
ceval-operating_system                          1c2571     accuracy       gen                          36.84
ceval-computer_architecture                     a74dad     accuracy       gen                          47.62
ceval-college_programming                       4ca32a     accuracy       gen                          54.05
ceval-college_physics                           963fa8     accuracy       gen                          36.84
ceval-college_chemistry                         e78857     accuracy       gen                          45.83
ceval-advanced_mathematics                      ce03e2     accuracy       gen                          31.58
ceval-probability_and_statistics                65e812     accuracy       gen                          38.89
ceval-discrete_mathematics                      e894ae     accuracy       gen                          31.25
ceval-electrical_engineer                       ae42b9     accuracy       gen                          29.73
ceval-metrology_engineer                        ee34ea     accuracy       gen                          75.00
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                          11.11
ceval-high_school_physics                       adf25f     accuracy       gen                          47.37
ceval-high_school_chemistry                     2ed27f     accuracy       gen                          47.37
ceval-high_school_biology                       8e2b9a     accuracy       gen                          57.89
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                          47.37
ceval-middle_school_biology                     86817c     accuracy       gen                          76.19
ceval-middle_school_physics                     8accf6     accuracy       gen                          84.21
ceval-middle_school_chemistry                   167a15     accuracy       gen                          80.00
ceval-veterinary_medicine                       b4e08d     accuracy       gen                          65.22
ceval-college_economics                         f3f4e6     accuracy       gen                          54.55
ceval-business_administration                   c1614e     accuracy       gen                          72.73
ceval-marxism                                   cf874c     accuracy       gen                          68.42
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                          79.17
ceval-education_science                         591fee     accuracy       gen                          75.86
ceval-teacher_qualification                     4e4ced     accuracy       gen                          72.73
ceval-high_school_politics                      5c0de2     accuracy       gen                          78.95
ceval-high_school_geography                     865461     accuracy       gen                          73.68
ceval-middle_school_politics                    5be3e7     accuracy       gen                          76.19
ceval-middle_school_geography                   8a63be     accuracy       gen                          83.33
ceval-modern_chinese_history                    fc01af     accuracy       gen                          73.91
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                          89.47
ceval-logic                                     f5b022     accuracy       gen                          40.91
ceval-law                                       a110a1     accuracy       gen                          37.50
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                          52.17
ceval-art_studies                               2a1300     accuracy       gen                          60.61
ceval-professional_tour_guide                   4e673e     accuracy       gen                          55.17
ceval-legal_professional                        ce8787     accuracy       gen                          43.48
ceval-high_school_chinese                       315705     accuracy       gen                          42.11
ceval-high_school_history                       7eb30a     accuracy       gen                          65.00
ceval-middle_school_history                     48ab4a     accuracy       gen                          86.36
ceval-civil_servant                             87d061     accuracy       gen                          51.06
ceval-sports_science                            70f27b     accuracy       gen                          52.63
ceval-plant_protection                          8941f9     accuracy       gen                          45.45
ceval-basic_medicine                            c409d6     accuracy       gen                          73.68
ceval-clinical_medicine                         49e82d     accuracy       gen                          50.00
ceval-urban_and_rural_planner                   95b885     accuracy       gen                          45.65
ceval-accountant                                002837     accuracy       gen                          61.22
ceval-fire_engineer                             bc23f5     accuracy       gen                          38.71
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                          54.84
ceval-tax_accountant                            3a5e3c     accuracy       gen                          61.22
ceval-physician                                 6e277d     accuracy       gen                          55.10
ceval-stem                                      -          naive_average  gen                          49.59
ceval-social-science                            -          naive_average  gen                          73.56
ceval-humanities                                -          naive_average  gen                          58.79
ceval-other                                     -          naive_average  gen                          53.60
ceval-hard                                      -          naive_average  gen                          36.28
ceval                                           -          naive_average  gen                          56.99
```

# 使用 OpenCompass 进行调用API评测

1. 使用 lmdeploy 启动 openai 兼容 api

```sh
lmdeploy serve api_server \
     models/internlm2_5-7b-chat \
     --backend turbomind \
     --model-format hf \
     --tp 1 \
     --cache-max-entry-count 0.8 \
     --quant-policy 0 \
     --model-name internlm2 \
     --server-name 0.0.0.0 \
     --server-port 23333
```

```sh
(lmdeploy) root@intern-studio-030876:~/opencompass/models# lmdeploy serve api_server \
>     internlm2_5-1_8b-chat \
>     --backend turbomind \
>     --model-format hf \
>     --tp 1 \
>     --cache-max-entry-count 0.8 \
>     --quant-policy 0 \
>     --model-name internlm2_5-1_8b-chat \
>     --server-name 0.0.0.0 \
>     --server-port 23333
[WARNING] gemm_config.in is not found; using default GEMM algo                                                                 
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
INFO:     Started server process [14898]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
```



