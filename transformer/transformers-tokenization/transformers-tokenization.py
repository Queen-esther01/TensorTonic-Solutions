import re
import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.id_to_word = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.bos_token,
            3: self.eos_token
        }
        
        result = []
        for text in texts:
            result.extend(text.split())
        result = sorted(set(result))

        for index, token in enumerate(result, len(self.word_to_id.keys())):
            self.word_to_id[token] = index
            self.id_to_word[index] = token

        self.vocab_size = len(self.word_to_id.keys())
            
        print(self.word_to_id)
        print(self.id_to_word)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        result = []
        split_text = text.split()
        # print(f'split {split_text}')
        for token in split_text:
            if token.lower() not in self.word_to_id:
                result.append(self.word_to_id[self.unk_token])
            else:
                result.append(self.word_to_id[token.lower()])
        # print(f'result: {result}')
        return result
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        print(f"ids: {ids}")
        text = " ".join([self.id_to_word[index] if index in self.id_to_word else '<UNK>' for index in ids ])
        print(f"decoded: {text}")
        return text
