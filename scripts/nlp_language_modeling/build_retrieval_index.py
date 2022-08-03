# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is the script to build Faiss retrieval index for KNN look up.
For more information about Faiss, check https://faiss.ai/

It requires the retrieval DB text data to be converted into `bin` and `idx` files by `preprocess_data_for_megatron.py` script.


Here is an example to using it:

```python
python scripts/nlp_language_modeling/build_retrieval_index.py \
    --input_file=PATH_TO_DB_FILE \
    --tokenizer-library=sentencepiece \
    --tokenizer-model=tokenizer.model \
    --train_index_size=128000 \
    --train_chunk_size=51200 \
    --devices=0,1,2,3 \
    --batch_size=1280 \
    --output_file=index.sav
```

It creates a index.sav which can be loaded by Faiss. It can look up the KNN chunk ids of the 
DB dataset given the input embedding vector. 

To use it in multiple stages, it follows the example as shown in  
https://github.com/facebookresearch/faiss/blob/main/demos/demo_ondisk_ivf.py

stage-0: train on the dataset, example,

```python
python scripts/nlp_language_modeling/build_retrieval_index.py \
    --input_file=PATH_TO_DB_FILE \
    --tokenizer-library=sentencepiece \
    --tokenizer-model=tokenizer.model \
    --train_index_size=128000 \
    --devices=0,1,2,3 \
    --stage=0 \
    --output_file=index_trained.sav
```

stage-1: build partial indexes, each containing a fraction of the dataset. This can be done in parallel on several machines

stage-2: merge the 4 indexes into one that is written directly to disk (needs not to fit in RAM)
"""
import argparse
import multiprocessing
from multiprocessing import Pool
from typing import Union

import faiss
from sentence_transformers import SentenceTransformer

from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import MMapRetrievalIndexedDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging
import numpy as np
import sys

QUEUE_SIZE = 30

queue = multiprocessing.Queue(QUEUE_SIZE)
emb_queue = multiprocessing.Queue(QUEUE_SIZE)


def get_tokenizer(args):
    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
        vocab_file=args.vocab_file,
        merges_file=args.merge_file,
        delimiter=args.delimiter,
    )
    if not hasattr(tokenizer, "pad_id"):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    elif hasattr(tokenizer, "pad_id") and (tokenizer.pad_id is None or tokenizer.pad_id < 0):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer


def process_sentence_chunks(
    ds: MMapRetrievalIndexedDataset, 
    tokenizer, 
    chunk_size: int, 
    warm_up_size: int, 
    percent: float, 
    stage: Union[int, None],
    workers: int,
):
    total_chunks = ds.chunks
    num_docs = len(ds._index.sizes)
    assert len(ds._index.sizes) == len(ds._index._chunk_id_start)
    if percent < 1.0:
        use_num_docs = int(num_docs * percent)
        logging.info(f"Use {use_num_docs} out of {num_docs} docs to build index")
        total_chunks = ds._index._chunk_id_start[min(use_num_docs, num_docs - 1)]
    logging.info(f"{total_chunks} chunks are used to build the index")

    if stage is None or stage == 0:
        # only prepare the warmup batch for stage None and stage 0
        assert warm_up_size < total_chunks
        warm_chunk_ids = np.random.randint(0, total_chunks,  warm_up_size)
        warm_up_slices = []
        for warm_up_id in warm_chunk_ids:
            warm_up_slices.append(ds.get_chunk(warm_up_id, force_no_cont_ids=True))
        with Pool(workers) as p:
            sentences = p.map(tokenizer.ids_to_text, warm_up_slices)
        queue.put(sentences)
        if stage == 0:
            # first the task for stage 0
            queue.put(None)

    start = 0
    threshold = 0.1
    while start < total_chunks:
        if start / total_chunks > threshold:
            logging.info(f"sentence processing {start / total_chunks} is done")
            threshold += 0.1
        id_slices = ds.get_chunk(slice(start, min(start + chunk_size, total_chunks)), force_no_cont_ids=True)
        start = min(start + chunk_size, total_chunks)
        sentences = [tokenizer.ids_to_text(ids) for ids in id_slices]
        queue.put(sentences)
    queue.put(None)


def get_sentence_chunks():
    return queue.get()


def calculate_embedding(pool, batch_size):
    while True:
        sentences = get_sentence_chunks()
        if sentences is None:
            break
        emb = model.encode_multi_process(sentences=sentences, pool=pool, batch_size=batch_size)
        emb_queue.put(emb)
    emb_queue.put(None)


def get_emb():
    return emb_queue.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build Faiss index",)
    parser.add_argument(
        '--input_file', type=str, required=True, help='Input file',
    )
    parser.add_argument(
        '--train_index_size', type=int, required=True, help='The number of sentences that is used to train the index',
    )
    parser.add_argument(
        '--train_chunk_size', type=int, default=10000, help='The sentences in chunks that is added to the index',
    )
    parser.add_argument(
        '--sentence_transformer_model',
        type=str,
        default='bert-base-nli-mean-tokens',
        help='sentence transformer to load',
    )
    parser.add_argument(
        '--output_file', type=str, required=True, help='Output Faiss index file',
    )
    parser.add_argument(
        '--percent', type=float, default=1.0, help='percent of documents used for building the search index',
    )
    parser.add_argument(
        '--devices', type=str, default=None, help='delimited list input with cuda devices. Specify like 0,1,2'
    )
    parser.add_argument(
        "--batch_size", type=int, default=4000, help="Batch size for encoding. Use max according to GPU MEM"
    )
    parser.add_argument("--subquantizers", type=int, default=8, help="Quantizer code size")
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--tokenizer-library',
        type=str,
        required=True,
        choices=['yttm', 'sentencepiece', 'megatron', 'huggingface', 'tabular'],
        help='What tokenizer library to use.',
    )
    group.add_argument(
        '--tokenizer-type', type=str, default=None, help='What type of tokenizer to use.',
    )
    group.add_argument(
        '--tokenizer-model', type=str, default=None, help='Path to tokenizer model.',
    )
    group.add_argument('--vocab-file', type=str, default=None, help='Path to the vocab file')
    group.add_argument('--workers', type=int, default=None, help='number of workers to run tokenizer')
    group.add_argument('--stage', type=int, default=None, help='used for building the large index in multiple stages', choices=[0, 1, 2])
    group.add_argument('--learned_index', type=str, default=None, help='the learned faiss index file, which is prepared at stage 0', choices=[0, 1, 2])
    group.add_argument('--merge-file', type=str, default=None, help='Path to the BPE merge file (if necessary).')
    group.add_argument('--delimiter', type=str, default=None, help='delimiter used for tabular tokenizer')

    args = parser.parse_args()
    model = SentenceTransformer(args.sentence_transformer_model)
    tokenizer = get_tokenizer(args)
    ds = MMapRetrievalIndexedDataset(args.input_file, skip_warmup=True)
    # make sure the dataset is padded as retrieval database
    assert ds._index.retrieval_db
    if ds.chunks < args.train_index_size:
        raise ValueError(
            f"the train index size {args.train_index_size} is larger than the total number of chunks {ds.chunks} in the dataset"
        )
    if args.stage is None or args.stage == 0:
        # Where nlist is 4*sqrt(N) to 16*sqrt(N), with N the size of the dataset. 
        # This just clusters the vectors with k-means. You will need between 30*K and 256*K vectors for training (the more the better).
        total_chunks = ds.chunks
        if args.percent < 1.0:
            num_docs = len(ds._index.sizes)
            use_num_docs = int(num_docs * args.percent)
            total_chunks = ds._index._chunk_id_start[min(use_num_docs, num_docs - 1)]
        nlist = int(8 * np.sqrt(total_chunks))
        assert 30 * nlist < args.train_index_size, f"need more training samples, at least {30 * nlist}"

    process = multiprocessing.Process(
        target=process_sentence_chunks,
        args=(ds, tokenizer, args.train_chunk_size, args.train_index_size, args.percent, args.stage, args.workers),
    )
    process.start()

    if args.devices is None:
        device_list = None
    else:
        device_list = ['cuda:' + str(device) for device in args.devices.split(',')]

    pool = model.start_multi_process_pool(device_list)

    emb_process = multiprocessing.Process(target=calculate_embedding, args=(pool, args.batch_size))
    emb_process.start()

    # get first batch of sentences to build up the index
    # sentences = get_sentence_chunks()

    emb = get_emb()

    if args.stage is None or args.stage == 0:
        # initialize the Faiss index
        # m is number of subquantizers. So vector of size D is broken into m sub-vectors of size D/m
        m = args.subquantizers
        k = 4  # num_nearest neighbors to get
        # gpu_resource = faiss.StandardGpuResources()
        # quantizer = faiss.GpuIndexFlatIP(gpu_resource, emb.shape[1])
        # index = faiss.GPUIndexIVFPQ(quantizer, emb.shape[1], nlist, m, 8)
        quantizer = faiss.IndexFlatIP(emb.shape[1])
        index = faiss.IndexIVFPQ(quantizer, emb.shape[1], nlist, m, 8)
    else:
        # stage 1 and stage 2, need to load the index from file
        index = faiss.read_index(args.learned_index)

    # 8 specifies that each sub-vector is encoded as 8 bits
    # build the index
    index.train(emb)
    logging.info('Trained Index')
    if args.stage is not None:
        logging.info('build index at stage {args.stage}')
    if args.stage == 0:
        # just need to have the learned index
        faiss.write_index(index, args.output_file)
        sys.exit(0)

    # add the first batch to the index
    index.add(emb)

    while True:
        emb = get_emb()
        if emb is None:
            break
        index.add(emb)
    process.join()
    emb_process.join()
    logging.info('Writing Index file')
    faiss.write_index(index, args.output_file)
    logging.info(f'Size of Index : {index.ntotal}')
    model.stop_multi_process_pool(pool)
