import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    """
    Base Dataset Loader

    Parameters
    ----------
    tokenizer : transformer_lens.HookedTokenizer.HookedTransformer.method
        Tokenizer to be used for tokenizing the text
    max_token_length : int
        Maximum length of the tokenized text
    padding_side : str
        Side to pad the tokens on, choice = ["right", "left"]
    last_token : bool
        Whether to give the index corresponding to the last token in the tokenized text
    add_question_structure : bool
        Whether to add a question framing to the text
    """

    def __init__(
        self,
        tokenizer,
        detokenizer,
        max_token_length=512,
        padding_side="right",
        last_token=False,
        add_question_structure=False,
        icl_few_shot=0,
    ):
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.last_token = last_token
        self.padding_side = padding_side
        self.max_token_length = max_token_length
        self.add_question_structure = add_question_structure
        self.pad_idx = self._get_pad_idx()

        self.pre_questionnaire_tokens = None
        self.post_questionnaire_tokens = None
        self.icl_few_shot = icl_few_shot

    def _pad_tokens(self, tokenized_text):
        # Get the length of the tokenized text
        tokenized_text_length = tokenized_text.shape[-1]

        # Get the total length of the tokenized text if the question structure tokens are added
        if self.add_question_structure:
            tokenized_text_length += (
                self.pre_questionnaire_tokens.shape[-1]
                + self.post_questionnaire_tokens.shape[-1]
            )

        # If the tokenized text is less than the max token length, pad it
        if tokenized_text_length < self.max_token_length:
            # Add the question structure tokens if required
            if self.add_question_structure:
                tokenized_text = torch.cat(
                    [
                        self.pre_questionnaire_tokens,
                        tokenized_text,
                        self.post_questionnaire_tokens,
                    ],
                    dim=-1,
                )

            # Get the index of the last token in the tokenized text
            last_token_index = tokenized_text.shape[-1] - 1

            # Padding the tokenized text
            pad_length = self.max_token_length - tokenized_text_length
            pad = torch.full(
                (tokenized_text.shape[0], pad_length),
                self.pad_idx,
                dtype=torch.long,
            ).to(tokenized_text.device)

            if self.padding_side == "right":
                tokenized_text = torch.cat((tokenized_text, pad), dim=-1)
            elif self.padding_side == "left":
                tokenized_text = torch.cat((pad, tokenized_text), dim=-1)
            else:
                raise ValueError(
                    f"Padding side must be either 'right' or 'left', got {self.padding_side}"
                )
        else:
            # If the tokenized text is greater than the max token length, truncate it
            # without adding the question structure tokens
            if self.add_question_structure:
                tokenized_text = tokenized_text[
                    :,
                    : self.max_token_length
                    - self.pre_questionnaire_tokens.shape[-1]
                    - self.post_questionnaire_tokens.shape[-1],
                ]

                tokenized_text = torch.cat(
                    [
                        self.pre_questionnaire_tokens,
                        tokenized_text,
                        self.post_questionnaire_tokens,
                    ],
                    dim=-1,
                )
            else:
                tokenized_text = tokenized_text[:, : self.max_token_length]
            last_token_index = self.max_token_length - 1

        return tokenized_text, last_token_index

    def _get_pad_idx(self):
        """
        Get the index of the padding token from the tokenizer vocab
        """

        random_text = [
            "As the man looked at the horizon, he realised that he had finally made it.",
            "Pizza has pepperoni on it.",
        ]

        tokens = self.tokenizer(random_text)
        self.pad_idx = tokens[-1][-1].item()
        return self.pad_idx

    def _get_option_indices(self, num_choices):
        """
        Get the indices of the options in the vocabulary of the dataset
        and tokenizer
        """

        option_indices = []
        for i in range(num_choices):
            option_indices.append(self.tokenizer(f" {chr(65+i)}")[-1][-1].item())

        return option_indices

    def _get_question_structure_tokens(self, pre_text, choices):
        """
        Add question structure to the text
        """

        pre_questionnaire_text = pre_text
        post_questionnaire_text = "\n"
        for i, choice in enumerate(choices):
            post_questionnaire_text += f"{chr(65+i)}: {choice}\n"
        post_questionnaire_text += "Answer:"

        pre_questionnaire_tokens = self.tokenizer(pre_questionnaire_text)
        post_questionnaire_tokens = self.tokenizer(post_questionnaire_text)

        # Remove the first padding/BOS token from the post_questionnaire_tokens as it is not required
        # because of direct concatenation
        post_questionnaire_tokens = post_questionnaire_tokens[:, 1:]

        return pre_questionnaire_tokens, post_questionnaire_tokens


class BaseAccuracyTracker(object):
    """
    Tracks the accuracy of the model
    """

    def __init__(
        self,
        option_indices,
        possible_targets,  # In order of A, B, C, D
    ):
        self.option_indices = option_indices
        self.possible_targets = {}
        for i, t in enumerate(possible_targets):
            self.possible_targets[i] = t

        self.total_per_class = {}
        self.correct_per_class = {}

        for t in self.possible_targets:
            self.total_per_class[t] = 0
            self.correct_per_class[t] = 0

    def update(self, scores, targets):
        """
        Update the accuracy tracker with the scores and targets
        """

        scores = scores.detach().cpu()
        scores = scores[:, self.option_indices]
        predictions = torch.argmax(scores, dim=1)
        targets = targets.detach().cpu()

        for key in self.possible_targets.keys():
            t = self.possible_targets[key]
            class_indices = torch.where(targets == t)[0]
            self.total_per_class[t] += len(class_indices)
            self.correct_per_class[t] += torch.sum(
                predictions[class_indices] == key
            ).item()

        return predictions.tolist()

    def print_accuracies(self):
        """
        Print the accuracies
        """

        total = 0
        correct = 0
        for t in self.total_per_class.keys():
            total += self.total_per_class[t]
            correct += self.correct_per_class[t]
            if self.total_per_class[t] == 0:
                print(f"No samples for class {self.possible_targets[t]}")
            else:
                print(
                    f"Accuracy for class {self.possible_targets[t]}: {self.correct_per_class[t]/self.total_per_class[t]:.2f}"
                )

        if total == 0:
            print("No samples in the dataset")
        else:
            print(f"Total Accuracy: {correct/total:.2f}")

    def save(self, file_path):
        """
        Save the accuracies
        """

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if sum(self.total_per_class.values()) == 0:
            net_accuracy = 0
        else:
            net_accuracy = sum(self.correct_per_class.values()) / sum(
                self.total_per_class.values()
            )

        with open(file_path, "w") as f:
            json.dump(
                {
                    "total_per_class": self.total_per_class,
                    "correct_per_class": self.correct_per_class,
                    "net_accuracy": net_accuracy,
                },
                f,
            )
