import torch
from datasets import load_dataset
from string import ascii_uppercase
from .base import BaseDataset, BaseAccuracyTracker


class CopaDataset(BaseDataset):
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
        if split == "train" or split == "test":
            if split == "test":
                raise ValueError(
                    "labels for the Test split not available for COPA dataset"
                )

            self.ds = load_dataset(
                "pkavumba/balanced-copa", split=split
            ) 
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
        pre_text = "Following are some multiple choice questions. You should directly answer the question by choosing the correct option.\n"
        self.pre_text = self._few_shot_pre_text(pre_text, self.icl_few_shot)

    def get_accuracy_tracker(self):
        return BaseAccuracyTracker(self._get_option_indices(2), [0, 1])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        premise = self.ds[idx]["premise"]
        choice1 = self.ds[idx]["choice1"]
        choice2 = self.ds[idx]["choice2"]
        question = self.ds[idx]["question"]

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
            return (tokenized_text, self.ds[idx]["label"]), last_token_index
        else:
            return tokenized_text, self.ds[idx]["label"]

    def _few_shot_pre_text(self, pre_text, num_examples):
        """
        Get few shot examples from the dataset
        """

        if num_examples != 0:
            indices = torch.randperm(len(self.ds))[:num_examples].tolist()
            for idx in indices:
                premise = self.ds[idx]["premise"]
                choice1 = self.ds[idx]["choice1"]
                choice2 = self.ds[idx]["choice2"]
                question = self.ds[idx]["question"]
                label = self.ds[idx]["label"]

                pre_text += self.prompt_templates["template1"]["with_options"](
                    premise, [choice1, choice2], question
                )
                pre_text += f" {ascii_uppercase[label]}\n"

        return pre_text
