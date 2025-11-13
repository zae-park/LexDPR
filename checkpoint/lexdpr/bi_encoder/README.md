---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:15427
- loss:MultipleNegativesRankingLoss
base_model: jhgan/ko-sroberta-multitask
widget:
- source_sentence: 'Represent this sentence for searching relevant passages: 상법 제651조의
    내용은 무엇인가?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 제1조(목적) 이 영은 「개인정보
    보호법」에서 위임된 사항과 그 시행에 필요한 사항을 규정함을 목적으로 한다.'
  - 'Represent this sentence for retrieving relevant passages: 제651조의2(서면에 의한 질문의
    효력) 보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다.'
  - 'Represent this sentence for retrieving relevant passages: 3손해배상액의 예정은 이행의 청구나
    계약의 해제에 영향을 미치지 아니한다.'
- source_sentence: 'Represent this sentence for searching relevant passages: 민법 제431조의
    내용은 무엇인가?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 1법원은 제62조제4항의 규정에 의한
    결정을 한 때에는 그 결정서를 관리인에게 송달하고 그 결정의 요지를 기재한 서면을 주주에게 송달하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 제866조(입양을 할 능력) 성년이
    된 사람은 입양(入養)을 할 수 있다.'
  - 'Represent this sentence for retrieving relevant passages: 1채무자가 보증인을 세울 의무가 있는
    경우에는 그 보증인은 행위능력 및 변제자력이 있는 자로 하여야 한다.'
- source_sentence: 'Represent this sentence for searching relevant passages: 유치권의
    성립요건인 유치권자의 점유에 간접점유가 포함되는지 여부(적극) 및 간접점유에서 점유매개관계를 이루는 임대차계약 등이 종료된 이후에도 직접점유자가
    목적물을 점유한 채 이를 반환하지 않고 있는 경우, 점유매개관계가 단절되는지 여부(소극)에 대한 법적 판단은?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 제194조(간접점유) 지상권, 전세권,
    질권, 사용대차, 임대차, 임치 기타의 관계로 타인으로 하여금 물건을 점유하게 한 자는 간접으로 점유권이 있다.'
  - 'Represent this sentence for retrieving relevant passages: 제709조(업무집행자의 대리권추정)
    조합의 업무를 집행하는 조합원은 그 업무집행의 대리권있는 것으로 추정한다.'
  - 'Represent this sentence for retrieving relevant passages: 1쌍무계약의 당사자 일방은 상대방이
    그 채무이행을 제공할 때 까지 자기의 채무이행을 거절할 수 있다. 그러나 상대방의 채무가 변제기에 있지 아니하는 때에는 그러하지 아니하다.'
- source_sentence: 'Represent this sentence for searching relevant passages: 표시·광고의
    공정화에 관한 법률상 허위·과장광고로 인한 손해배상청구권을 가지고 있던 아파트 수분양자가 수분양자의 지위를 제3자에게 양도한 경우, 양수인이
    당연히 위 손해배상청구권을 행사할 수 있는지 여부(소극) 및 양수인이 위 손해배상청구권을 행사할 수 있는 경우에 대한 법적 판단은?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 제105조(임의규정) 법률행위의 당사자가
    법령 중의 선량한 풍속 기타 사회질서에 관계없는 규정과 다른 의사를 표시한 때에는 그 의사에 의한다.'
  - 'Represent this sentence for retrieving relevant passages: 제106조(사실인 관습) 법령 중의
    선량한 풍속 기타 사회질서에 관계없는 규정과 다른 관습이 있는 경우에 당사자의 의사가 명확하지 아니한 때에는 그 관습에 의한다.'
  - 'Represent this sentence for retrieving relevant passages: 1지명채권의 양도는 양도인이 채무자에게
    통지하거나 채무자가 승낙하지 아니하면 채무자 기타 제삼자에게 대항하지 못한다.'
- source_sentence: 'Represent this sentence for searching relevant passages: 채무자에
    대한 파산선고 후 파산채권자가 채권자취소의 소를 제기할 수 있는지 여부(소극)에 대한 법적 판단은?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 2개인인 채무자 또는 개인이 아닌
    채무자의 이사는 제1항에 규정에 의한 관리인의 권한을 침해하거나 부당하게 그 행사에 관여할 수 없다.'
  - 'Represent this sentence for retrieving relevant passages: 3. 피상속인의 형제자매'
  - 'Represent this sentence for retrieving relevant passages: 1파산재단에 속하는 재산에 관하여
    파산선고 당시 법원에 계속되어 있는 소송은 파산관재인 또는 상대방이 이를 수계할 수 있다. 제335조제1항의 규정에 의하여 파산관재인이 채무를
    이행하는 경우에 상대방이 가지는 청구권에 관한 소송의 경우에도 또한 같다.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on jhgan/ko-sroberta-multitask

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask) <!-- at revision ab957ae6a91e99c4cad36d52063a2a9cf1bf4419 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: PeftModelForFeatureExtraction 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Represent this sentence for searching relevant passages: 채무자에 대한 파산선고 후 파산채권자가 채권자취소의 소를 제기할 수 있는지 여부(소극)에 대한 법적 판단은?',
    'Represent this sentence for retrieving relevant passages: 1파산재단에 속하는 재산에 관하여 파산선고 당시 법원에 계속되어 있는 소송은 파산관재인 또는 상대방이 이를 수계할 수 있다. 제335조제1항의 규정에 의하여 파산관재인이 채무를 이행하는 경우에 상대방이 가지는 청구권에 관한 소송의 경우에도 또한 같다.',
    'Represent this sentence for retrieving relevant passages: 2개인인 채무자 또는 개인이 아닌 채무자의 이사는 제1항에 규정에 의한 관리인의 권한을 침해하거나 부당하게 그 행사에 관여할 수 없다.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 15,427 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              |
  | details | <ul><li>min: 31 tokens</li><li>mean: 76.84 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 26 tokens</li><li>mean: 68.57 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                      |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: 독점규제 및 공정거래에 관한 법률 시행령 제94조의 내용은 무엇인가?</code>                                                                                                                | <code>Represent this sentence for retrieving relevant passages: 제94조(과태료의 부과기준) 법 제130조에 따른 과태료의 부과기준은 다음 각 호의 구분에 따른다.</code>                                                                                                                                                  |
  | <code>Represent this sentence for searching relevant passages: 채무자 회생 및 파산에 관한 법률 제477조의 내용은 무엇인가?</code>                                                                                                                   | <code>Represent this sentence for retrieving relevant passages: 1파산재단이 재단채권의 총액을 변제하기에 부족한 것이 분명하게 된 때에는 재단채권의 변제는 다른 법령이 규정하는 우선권에 불구하고 아직 변제하지 아니한 채권액의 비율에 따라 한다. 다만, 재단채권에 관하여 존재하는 유치권ᆞ질권ᆞ저당권ᆞ「동산ᆞ채권 등의 담보에 관한 법률」에 따른 담보권 및 전세권의 효력에는 영향을 미치지 아니한다. <개정 2010.6.10></code> |
  | <code>Represent this sentence for searching relevant passages: 토지와 함께 공동근저당권이 설정된 건물이 그대로 존속함에도 등기부에 멸실의 기재가 이루어지고 이를 이유로 등기부가 폐쇄된 후 토지에 대하여만 경매절차가 진행되어 토지와 건물의 소유자가 달라진 경우, 건물을 위한 법정지상권이 성립하는지 여부(적극)에 대한 법적 판단은?</code> | <code>Represent this sentence for retrieving relevant passages: 제366조(법정지상권) 저당물의 경매로 인하여 토지와 그 지상건물이 다른 소유자에 속한 경우에는 토지소유자는 건물소유자에 대하여 지상권을 설정한 것으로 본다. 그러나 지료는 당사자의 청구에 의하여 법원이 이를 정한다.</code>                                                                                |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 0.05,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.1296 | 500  | 1.3812        |
| 0.2593 | 1000 | 1.3744        |
| 0.3889 | 1500 | 1.3719        |
| 0.5185 | 2000 | 1.3702        |
| 0.6482 | 2500 | 1.3686        |
| 0.7778 | 3000 | 1.3685        |
| 0.9074 | 3500 | 1.3681        |


### Framework Versions
- Python: 3.12.5
- Sentence Transformers: 3.1.1
- Transformers: 4.43.4
- PyTorch: 2.5.1+cu121
- Accelerate: 0.34.2
- Datasets: 2.21.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->