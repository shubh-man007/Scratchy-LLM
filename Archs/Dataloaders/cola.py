import torch
from datasets import load_dataset
from .base import BaseDataset, BaseAccuracyTracker


class ColaDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        detokenizer,
        max_token_length=512,
        padding_side="right",
        split="test",
        last_token=False,
        add_question_structure=False,
        icl_few_shot=0,
    ):

        super().__init__(
            tokenizer,
            detokenizer,
            max_token_length,
            padding_side,
            last_token,
            add_question_structure,
            icl_few_shot,
        )
        if split == "train" or split == "validation" or split == "test":
            self.ds = load_dataset("nyu-mll/glue", "cola", split=split)
        else:
            raise ValueError("Invalid split")

        pre_text = "Following are some multiple choice questions. You should directly answer the question by choosing the correct option.\n"
        choices = ["Unacceptable", "Acceptable"]

        pre_text = self._few_shot_pre_text(pre_text, choices, self.icl_few_shot)
        self.pre_questionnaire_tokens, self.post_questionnaire_tokens = (
            self._get_question_structure_tokens(pre_text=pre_text, choices=choices)
        )

    def get_accuracy_tracker(self):
        return BaseAccuracyTracker(self._get_option_indices(2), [0, 1])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["sentence"]
        tokenized_text = self.tokenizer(text)

        if self.add_question_structure:
            tokenized_text = tokenized_text[:, 1:]

        tokenized_text, last_token_index = self._pad_tokens(tokenized_text)
        tokenized_text = tokenized_text.squeeze()

        if self.last_token:
            return (tokenized_text, self.ds[idx]["label"]), last_token_index
        else:
            return tokenized_text, self.ds[idx]["label"]

    def _few_shot_pre_text(self, pre_text, choices, num_examples):
        """
        Get few shot examples from the dataset
        """

        template = "Question: Select the suitable option for the following statement - "
        if num_examples != 0:
            indices = torch.randperm(len(self.ds))[:num_examples].tolist()
            for idx in indices:
                text = self.ds[idx]["sentence"]
                label = self.ds[idx]["label"]

                pre_text += f"{template}: {text}\n"
                for i, choice in enumerate(choices):
                    pre_text += f"{chr(65+i)}: {choice}\n"
                pre_text += f"Answer: {chr(65+label)}\n"

        pre_text += template
        return pre_text
