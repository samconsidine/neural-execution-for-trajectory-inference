from dataclasses import dataclass

from torch import Tensor
from typing import Optional, Any

@dataclass
class RNASeqDataset:
    X: Tensor
    y: Tensor
    info: Optional[Any] = None
