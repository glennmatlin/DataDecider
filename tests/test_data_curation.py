# tests/test_data_curation.py
import unittest
import tempfile
import json
import os
from olmo.data.data_curation import DataDecideCurator


class TestDataCuration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create sample JSON data
        sample_data = [
            {
                "url": "null",
                "shard": "test-0001.json",
                "text": "This is a test document for data curation.",
                "uuid": "test-uuid-1",
            },
            {
                "url": "null",
                "shard": "test-0001.json",
                "text": "Another test document with different content.",
                "uuid": "test-uuid-2",
            },
        ]

        with open(os.path.join(self.temp_dir, "test.json"), "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

    def test_load_json_data(self):
        curator = DataDecideCurator(self.temp_dir)
        data = curator.load_json_data()

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["uuid"], "test-uuid-1")

    def test_compute_statistics(self):
        curator = DataDecideCurator(self.temp_dir)
        data = curator.load_json_data()
        stats = curator.compute_data_statistics(data)

        self.assertIn("total_documents", stats)
        self.assertIn("avg_length", stats)
        self.assertEqual(stats["total_documents"], 2)


if __name__ == "__main__":
    unittest.main()
