import unittest
from utils import cached_dataset_loader, validate_inputs
import torch

class TestUtils(unittest.TestCase):
    def test_cached_dataset_loader(self):
        """Test the LRU cache for dataset loading."""
        dataloader = cached_dataset_loader("WSJ0-2mix")
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

    def test_validate_inputs(self):
        """Test the input validation decorator."""
        @validate_inputs(lambda x: x > 0)
        def dummy_function(x):
            return x

        self.assertEqual(dummy_function(5), 5)
        with self.assertRaises(ValueError):
            dummy_function(-1)

if __name__ == "__main__":
    unittest.main()