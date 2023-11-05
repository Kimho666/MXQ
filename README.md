# MXQ
Harware-friendly Mixed-precision 2-4 Quantization Method with QAT Fintune

![image](https://github.com/Kimho666/MXQ/assets/137678908/bff35113-1ba5-4cd0-8013-09e6ebc6b274)

![image](https://github.com/Kimho666/MXQ/assets/137678908/94b527ab-31c7-41ac-a868-5235fa779ab4)

使用说明：
一、finetune步骤

1. 进入LLM-QAT目录，直接运行 bash run_train.sh 2 32 32即可（2/4混合量化中4的部分已在代码中固定）

注：训练数据暂时未上传；2/4混合finetune的代码主要更新在/LLM-QAT/models/util_quant.py中，class MXAsymQuantizer.

二、量化步骤

1. 进入mxq_quant目录，直接运行 如下命令（model路径需修改）

python main.py --model /user/jhli/quantization/model/Llama-2-7b-hf --prune_method mxq

如果想保存model做后续评估

python main.py --model /user/jhli/quantization/model/Llama-2-7b-hf --prune_method mxq --save_model 路径

三、harness-eval步骤

1. cd mxq_quant/lm-evaluation-harness/

2. python setup.py install
   
4. cd ..
   
6. python lmeval.py --model hf-causal --model_args pretrained=保存的模型路径,dtype=float16,use_accelerate=True --tasks winogrande,piqa,hellaswag,arc_easy,wikitext



