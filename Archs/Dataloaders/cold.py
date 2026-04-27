import random
import numpy as np
import os
import torch
from datasets import load_dataset
from string import ascii_uppercase
from .base import BaseDataset, BaseAccuracyTracker
import pandas as pd


class ColdDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        detokenizer,
        max_token_length=512,
        padding_side="right",
        split="train",
        last_token=False,
        add_question_structure=False,
        icl_few_shot=0,
        activity="going grocery shopping",
        num_samples=1000,
        seed=42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        super().__init__(
            tokenizer,
            detokenizer,
            max_token_length,
            padding_side,
            last_token,
            add_question_structure,
            icl_few_shot,
        )
        if split == "train" or split == "test":
            if split == "test":
                raise ValueError(
                    "labels for the Test split not available for COPA dataset"
                )

            df = pd.read_csv(
                "custom_dataloaders/COLD_data/causal_query_triplets_sampled_instances/"
                + activity
                + ".csv"
            )
            df1 = df[df["label"] == 0].sample(num_samples // 2)
            df2 = df[df["label"] == 1].sample(num_samples // 2)
            self.ds = pd.concat([df1, df2], ignore_index=True)
            del df, df1, df2

        else:
            raise ValueError("Invalid split")

        self.prompt_templates = {
            "template1": {
                "with_options": lambda premise, choices, causal_question: f"Question: Which of the following events (given as options A or B) is a more plausible {causal_question} of the event '{premise}'?\n"
                + "\n".join(
                    [
                        f"{ascii_uppercase[i]}: {choice}"
                        for i, choice in enumerate(choices)
                    ]
                )
                + "\nAnswer:",
                "without_options": lambda premise, causal_question: f"Question: Which of the following events (given as options A or B) is a more plausible {causal_question} of the event '{premise}'?",
            }
        }
        pre_text = f"Following are some multiple choice questions about the activity '{activity}'. You should directly answer the question by choosing the correct option.\n"
        self.pre_text = self._few_shot_pre_text(pre_text, self.icl_few_shot)

    def get_accuracy_tracker(self):
        return BaseAccuracyTracker(self._get_option_indices(2), [0, 1])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        premise = self.ds["premise"][idx]
        choice1 = self.ds["choice1"][idx]
        choice2 = self.ds["choice2"][idx]
        question = self.ds["question"][idx]
        prompt = self.prompt_templates["template1"]["without_options"](
            premise, question
        )

        if self.add_question_structure:
            self.pre_questionnaire_tokens, self.post_questionnaire_tokens = (
                self._get_question_structure_tokens(
                    pre_text=self.pre_text, choices=[choice1, choice2]
                )
            )

        tokenized_text = self.tokenizer(prompt)

        if self.add_question_structure:
            tokenized_text = tokenized_text[:, 1:]

        tokenized_text, last_token_index = self._pad_tokens(tokenized_text)
        tokenized_text = tokenized_text.squeeze()

        if self.last_token:
            return (tokenized_text, self.ds["label"][idx]), last_token_index
        else:
            return tokenized_text, self.ds["label"][idx]

    def _few_shot_pre_text(self, pre_text, num_examples):
        """
        Get few shot examples from the dataset
        """

        if num_examples != 0:
            indices = torch.randperm(len(self.ds))[:num_examples].tolist()
            for idx in indices:
                premise = self.ds["premise"][idx]
                choice1 = self.ds["choice1"][idx]
                choice2 = self.ds["choice2"][idx]
                question = self.ds["question"][idx]
                label = self.ds["label"][idx]
                pre_text += self.prompt_templates["template1"]["with_options"](
                    premise, [choice1, choice2], question
                )
                pre_text += f" {ascii_uppercase[label]}\n"

        return pre_text
