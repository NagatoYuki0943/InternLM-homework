# 基础作业

## 使用 OpenCompass 评测 internlm2-chat-1_8b 模型在 C-Eval 数据集上的性能

### 概览



在 OpenCompass 中评估一个模型通常包括以下几个阶段：配置 -> 推理 -> 评估 -> 可视化。

- 配置：这是整个工作流的起点。您需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。
- 推理与评估：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。推理阶段主要是让模型从数据集产生输出，而评估阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。如果需要了解该问题及解决方案，可以参考 FAQ: 效率。
- 可视化：评估完成后，OpenCompass 将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件。你也可以激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告。 接下来，我们将展示 OpenCompass 的基础用法，展示书生浦语在 `C-Eval` 基准任务上的评估。它们的配置文件可以在 `configs/eval_demo.py` 中找到。

### 安装

```sh
cd /root

git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -v -e .

# 补充安装
pip install -r requirements.txt
pip install protobuf
```

![](InternLM2_homework7.assets/1_8b_ceval1.png)

### 数据准备

解压评测数据集到 data/ 处

```sh
cd /root/opencompass
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
rm OpenCompassData-core-20231110.zip
```

![](InternLM2_homework7.assets/1_8b_ceval2.png)

### 查看支持的数据集和模型

列出所有跟 internlm 及 ceval 相关的配置

```sh
cd /root/opencompass
python tools/list_configs.py internlm ceval
```

运行命令

```sh
(lm) root@intern-studio-030876:~/opencompass# python tools/list_configs.py internlm ceval
+----------------------------------------+----------------------------------------------------------------------+
| Model                                  | Config Path                                                          |
|----------------------------------------+----------------------------------------------------------------------|
| hf_internlm2_1_8b                      | configs/models/hf_internlm/hf_internlm2_1_8b.py                      |
| hf_internlm2_20b                       | configs/models/hf_internlm/hf_internlm2_20b.py                       |
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
| hf_internlm_20b                        | configs/models/hf_internlm/hf_internlm_20b.py                        |
| hf_internlm_7b                         | configs/models/hf_internlm/hf_internlm_7b.py                         |
| hf_internlm_chat_20b                   | configs/models/hf_internlm/hf_internlm_chat_20b.py                   |
| hf_internlm_chat_7b                    | configs/models/hf_internlm/hf_internlm_chat_7b.py                    |
| hf_internlm_chat_7b_8k                 | configs/models/hf_internlm/hf_internlm_chat_7b_8k.py                 |
| hf_internlm_chat_7b_v1_1               | configs/models/hf_internlm/hf_internlm_chat_7b_v1_1.py               |
| internlm_7b                            | configs/models/internlm/internlm_7b.py                               |
| lmdeploy_internlm2_chat_20b            | configs/models/hf_internlm/lmdeploy_internlm2_chat_20b.py            |
| lmdeploy_internlm2_chat_7b             | configs/models/hf_internlm/lmdeploy_internlm2_chat_7b.py             |
| ms_internlm_chat_7b_8k                 | configs/models/ms_internlm/ms_internlm_chat_7b_8k.py                 |
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
| ceval_ppl                      | configs/datasets/ceval/ceval_ppl.py                              |
| ceval_ppl_1cd8bf               | configs/datasets/ceval/ceval_ppl_1cd8bf.py                       |
| ceval_ppl_578f8d               | configs/datasets/ceval/ceval_ppl_578f8d.py                       |
| ceval_ppl_93e5ce               | configs/datasets/ceval/ceval_ppl_93e5ce.py                       |
| ceval_zero_shot_gen_bd40ef     | configs/datasets/ceval/ceval_zero_shot_gen_bd40ef.py             |
+--------------------------------+------------------------------------------------------------------+
```

![](InternLM2_homework7.assets/1_8b_ceval3.png)



### 启动评测 

确保按照上述步骤正确安装 OpenCompass 并准备好数据集后，可以通过以下命令评测 InternLM2-Chat-1.8B 模型在 C-Eval 数据集上的性能。由于 OpenCompass 默认并行启动评估过程，我们可以在第一次运行时以 --debug 模式启动评估，并检查是否存在问题。在 --debug 模式下，任务将按顺序执行，并实时打印输出。

```sh
cd /root/opencompass

export MKL_SERVICE_FORCE_INTEL=1
#或
export MKL_THREADING_LAYER=GNU

python run.py \
--datasets ceval_gen \
--hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace 模型路径
--tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 1024 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 2  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
--debug

```

运行命令

```sh
(opencompass) root@intern-studio-030876:~/opencompass# cd /root/opencompass
(opencompass) root@intern-studio-030876:~/opencompass# export MKL_SERVICE_FORCE_INTEL=1
(opencompass) root@intern-studio-030876:~/opencompass# python run.py \
> --datasets ceval_gen \
> --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
> --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
> --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
> --model-kwargs device_map='auto' trust_remote_code=True \
> --max-seq-len 1024 \
> --max-out-len 16 \
> --batch-size 2  \
> --num-gpus 1
04/19 19:22:32 - OpenCompass - INFO - Loading ceval_gen: configs/datasets/ceval/ceval_gen.py
04/19 19:22:32 - OpenCompass - INFO - Loading example: configs/summarizers/example.py
04/19 19:22:32 - OpenCompass - WARNING - SlurmRunner is not used, so the partition argument is ignored.
04/19 19:22:32 - OpenCompass - INFO - Partitioned into 1 tasks.
launch OpenICLInfer[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_economics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-accountant,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-tax_accountant,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-physician,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-civil_servant,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-urban_and_rural_planner,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-teacher_qualification,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_programming,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-electrical_engineer,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-business_administration,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-art_studies,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-fire_engineer,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-environmental_impact_assessment_engineer,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-education_science,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-professional_tour_guide,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_chemistry,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-metrology_engineer,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-mao_zedong_thought,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-law,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-veterinary_medicine,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-modern_chinese_history,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-chinese_language_and_literature,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-legal_professional,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-logic,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_history,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-plant_protection,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-clinical_medicine,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-computer_architecture,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_biology,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_politics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_chemistry,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_history,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-computer_network,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-operating_system,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_physics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-advanced_mathematics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_physics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_chemistry,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_biology,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_mathematics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_physics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-marxism,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_politics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_geography,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-ideological_and_moral_cultivation,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_chinese,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-sports_science,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-basic_medicine,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-probability_and_statistics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_mathematics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-discrete_mathematics,opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_geography] on GPU 0
100%|███████████████████████████████████████████████████████████| 1/1 [14:00<00:00, 840.07s/it]
04/19 19:36:33 - OpenCompass - INFO - Partitioned into 52 tasks.
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-computer_network] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-operating_system] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-computer_architecture] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_programming] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_physics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_chemistry] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-advanced_mathematics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-probability_and_statistics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-discrete_mathematics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-electrical_engineer] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-metrology_engineer] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_mathematics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_physics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_chemistry] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_biology] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_biology] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_mathematics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_physics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_chemistry] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-veterinary_medicine] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-college_economics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-business_administration] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-marxism] on CPU    
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-education_science] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-mao_zedong_thought] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-teacher_qualification] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_politics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_geography] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_politics] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_geography] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-modern_chinese_history] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-ideological_and_moral_cultivation] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-logic] on CPU      
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-law] on CPU        
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-professional_tour_guide] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-chinese_language_and_literature] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-art_studies] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-legal_professional] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_chinese] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-high_school_history] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-middle_school_history] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-civil_servant] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-sports_science] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-plant_protection] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-basic_medicine] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-clinical_medicine] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-urban_and_rural_planner] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-accountant] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-fire_engineer] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-environmental_impact_assessment_engineer] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-tax_accountant] on CPU 
launch OpenICLEval[opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-physician] on CPU  
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 52/52 [07:34<00:00,  8.74s/it]
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b
----------------------------------------------  ---------  -------------  ------  ---------------------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                                       47.37
ceval-operating_system                          1c2571     accuracy       gen                                                                                       47.37
ceval-computer_architecture                     a74dad     accuracy       gen                                                                                       23.81
ceval-college_programming                       4ca32a     accuracy       gen                                                                                       13.51
ceval-college_physics                           963fa8     accuracy       gen                                                                                       42.11
ceval-college_chemistry                         e78857     accuracy       gen                                                                                       33.33
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                                       10.53
ceval-probability_and_statistics                65e812     accuracy       gen                                                                                       38.89
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                                       25
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                                       27.03
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                                       54.17
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                                       16.67
ceval-high_school_physics                       adf25f     accuracy       gen                                                                                       42.11
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                                       47.37
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                                       26.32
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                                       36.84
ceval-middle_school_biology                     86817c     accuracy       gen                                                                                       80.95
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                                       47.37
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                                       80
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                                       43.48
ceval-college_economics                         f3f4e6     accuracy       gen                                                                                       32.73
ceval-business_administration                   c1614e     accuracy       gen                                                                                       36.36
ceval-marxism                                   cf874c     accuracy       gen                                                                                       68.42
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                                       70.83
ceval-education_science                         591fee     accuracy       gen                                                                                       55.17
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                                       59.09
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                                       57.89
ceval-high_school_geography                     865461     accuracy       gen                                                                                       47.37
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                                       71.43
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                                       75
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                                       52.17
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                                       73.68
ceval-logic                                     f5b022     accuracy       gen                                                                                       27.27
ceval-law                                       a110a1     accuracy       gen                                                                                       29.17
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                                       47.83
ceval-art_studies                               2a1300     accuracy       gen                                                                                       42.42
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                                       51.72
ceval-legal_professional                        ce8787     accuracy       gen                                                                                       34.78
ceval-high_school_chinese                       315705     accuracy       gen                                                                                       42.11
ceval-high_school_history                       7eb30a     accuracy       gen                                                                                       65
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                                       86.36
ceval-civil_servant                             87d061     accuracy       gen                                                                                       42.55
ceval-sports_science                            70f27b     accuracy       gen                                                                                       52.63
ceval-plant_protection                          8941f9     accuracy       gen                                                                                       40.91
ceval-basic_medicine                            c409d6     accuracy       gen                                                                                       68.42
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                                       31.82
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                                       47.83
ceval-accountant                                002837     accuracy       gen                                                                                       36.73
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                                       38.71
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                                       51.61
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                                       36.73
ceval-physician                                 6e277d     accuracy       gen                                                                                       42.86
ceval-stem                                      -          naive_average  gen                                                                                       39.21
ceval-social-science                            -          naive_average  gen                                                                                       57.43
ceval-humanities                                -          naive_average  gen                                                                                       50.23
ceval-other                                     -          naive_average  gen                                                                                       44.62
ceval-hard                                      -          naive_average  gen                                                                                       32
ceval                                           -          naive_average  gen                                                                                       46.19
04/19 19:44:07 - OpenCompass - INFO - write summary to /root/opencompass/outputs/default/20240419_192232/summary/summary_20240419_192232.txt
04/19 19:44:07 - OpenCompass - INFO - write csv to /root/opencompass/outputs/default/20240419_192232/summary/summary_20240419_192232.csv
```

![1_8b_ceval4](InternLM2_homework7.assets/1_8b_ceval4.png)

# 进阶作业



## 将自定义数据集提交至OpenCompass官网