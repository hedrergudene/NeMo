ds_path=/home/apeganov/NeMo/examples/speech_translation/punct_prepared_datasets/prepared_punctuation_data_min_punctuation_27.10.2021_3.13
output="${ds_path}/2370440_cross_labels_seq2seq_from_scratch_original_steps300000_gpu1/pred_cross_labels.txt"
model_path="NeMo/examples/nlp/machine_translation/ngc_results/aayn_base_min_punc_cross/2370440_from_scratch_original_steps300000_gpu1/nemo_experiments/AAYNBase/2021-10-26_22-33-24/checkpoints/AAYNBase.nemo"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model="${model_path}" \
    --srctext="${ds_path}/test/input.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \
    --word_tokens a b c d e f g h i j k l m n o p q \
    --add_src_num_words_to_batch
