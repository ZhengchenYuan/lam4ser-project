import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer_utils import print_tokenizer_diagnostics


if __name__ == "__main__":
    print_tokenizer_diagnostics()
