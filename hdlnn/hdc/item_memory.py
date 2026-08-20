import torch
import logging
from typing import Dict, Tuple, Set, List

logger = logging.getLogger(__name__)

class ItemMemory:
    """Stores basis hypervectors for categorical tokens.
    
    Optimized for fast batch retrieval via tensor indexing after locking.
    """
    def __init__(self, D: int = 10000):
        self.D = D
        self.locked = False
        # Mapping: categorical_column -> token_value -> token_id (int)
        self.vocab_ids: Dict[str, Dict[str, int]] = {}
        # List of tensors to be stacked at lock time: categorical_column -> List[torch.Tensor]
        # Index 0 is always reserved for OOV
        self.vocab_vectors_list: Dict[str, List[torch.Tensor]] = {}
        # Stacked 2D tensor of shape [vocab_size, D]: categorical_column -> torch.Tensor
        self.vocab_tensors: Dict[str, torch.Tensor] = {}

    def lock(self) -> None:
        """Locks vocabulary and compiles lists of hypervectors into 2D tensors for fast lookup."""
        self.locked = True
        for col in self.vocab_ids:
            vectors = self.vocab_vectors_list[col]
            self.vocab_tensors[col] = torch.stack(vectors)
        logger.info("Item Memory locked. Vocab compiled into 2D tensors for optimized batch processing.")

    def unlock(self) -> None:
        """Unlocks item memory for dynamic training registration."""
        self.locked = False
        self.vocab_tensors = {}
        logger.info("Item Memory unlocked. Ready for token registration.")

    def get_vector(self, key: str, value: str) -> Tuple[torch.Tensor, bool]:
        """Retrieves the hypervector for a single key-value pair.
        
        Compatible with single-flow processing, but slow for batch processing.
        """
        # Ensure category structures exist
        if key not in self.vocab_ids:
            self.vocab_ids[key] = {"<OOV>": 0}
            self.vocab_vectors_list[key] = [self._random_bipolar_vector()]
            if self.locked:
                self.vocab_tensors[key] = torch.stack(self.vocab_vectors_list[key])

        vocab_col = self.vocab_ids[key]
        if value in vocab_col:
            idx = vocab_col[value]
            is_oov = (idx == 0)
            if self.locked:
                return self.vocab_tensors[key][idx], is_oov
            else:
                return self.vocab_vectors_list[key][idx], is_oov
        
        if self.locked:
            # OOV case
            return self.vocab_tensors[key][0], True
        else:
            # Register new token
            new_vector = self._random_bipolar_vector()
            idx = len(vocab_col)
            vocab_col[value] = idx
            self.vocab_vectors_list[key].append(new_vector)
            return new_vector, False

    def get_batch_vectors(self, key: str, values: List[str]) -> Tuple[torch.Tensor, List[bool]]:
        """Optimized batch lookup returning a 2D tensor of shape [batch_size, D] and OOV flags."""
        if key not in self.vocab_ids:
            # Initialize empty category
            self.vocab_ids[key] = {"<OOV>": 0}
            self.vocab_vectors_list[key] = [self._random_bipolar_vector()]
            if self.locked:
                self.vocab_tensors[key] = torch.stack(self.vocab_vectors_list[key])

        vocab_col = self.vocab_ids[key]
        
        # Fast list comprehension mapping tokens to integer IDs (defaulting to 0 for OOV)
        indices = [vocab_col.get(val, 0) for val in values]
        is_oov = [idx == 0 for idx in indices]
        
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        if self.locked:
            val_tensor = self.vocab_tensors[key][indices_tensor]
        else:
            # Temporary fallback if called before locking
            tensors_list = self.vocab_vectors_list[key]
            val_tensor = torch.stack([tensors_list[idx] for idx in indices])
        
        return val_tensor, is_oov

    def _random_bipolar_vector(self) -> torch.Tensor:
        """Generate a random bipolar vector without requiring torchhd."""
        return torch.randint(0, 2, (self.D,), dtype=torch.int8).float() * 2.0 - 1.0
