import logging, traceback, torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator, CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator
from sentence_transformers.util import mine_hard_negatives

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def main():
    model_name = "Alibaba-NLP/gte-reranker-modernbert-base"
    train_batch_size = 128
    num_epochs = 6
    num_hard_negatives = 5

    model = CrossEncoder(
        model_name,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="gte-reranker-modernbert trained on buidlaiv01",
        ),
    )
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    logging.info("Load buidlaiv01 dataset")
    full_dataset = load_dataset("wryta/buidlaiv01", split="train")
    dataset_dict = full_dataset.train_test_split(test_size=1000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        num_negatives=num_hard_negatives,
        margin=0,
        range_min=0,
        range_max=100,
        sampling_strategy="top",
        batch_size=4096,
        output_format="labeled-pair",
        use_faiss=True,
    )
    logging.info(hard_train_dataset)

    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))
    nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=train_batch_size,
    )
    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        corpus=full_dataset["answer"],
        num_negatives=30,
        batch_size=4096,
        include_positives=True,
        output_format="n-tuple",
        use_faiss=True,
    )
    logging.info(hard_eval_dataset)
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=[
            {
                "query": sample["question"],
                "positive": [sample["answer"]],
                "documents": [sample[col] for col in hard_eval_dataset.column_names[2:]],
            }
            for sample in hard_eval_dataset
        ],
        batch_size=train_batch_size,
        name="buidlaiv01-dev",
        always_rerank_positives=False,
    )
    evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
    evaluator(model)

    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-buidlaiv01-bce"
    args = CrossEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_buidlaiv01-dev_ndcg@10",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        logging_first_step=True,
        run_name=run_name,
        seed=12,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=hard_train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    evaluator(model)
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(f"Error uploading model:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
