import json
from pathlib import Path
import tempfile
import unittest

from backend.predictions import PredictionStore


class PredictionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "predictions"
        self.store = PredictionStore(self.root)

    def test_persisted_prediction_survives_store_restart(self):
        payload = {"result": {"prediction": {"latency_seconds": 61.5}}, "model_scope": {"notes": ["参考硬件"]}}
        record = self.store.save(payload)
        self.assertRegex(record["id"], r"^prediction-[0-9a-f]{32}$")
        self.assertEqual(PredictionStore(self.root).get(record["id"]), record)
        self.assertNotIn("id", payload)
        self.assertEqual(record["result"], payload["result"])
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_ids_cannot_override_or_traverse_the_store(self):
        first = self.store.save({"id": "../private", "created_at": "spoofed"})
        second = self.store.save({"id": first["id"]})
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["created_at"], "spoofed")
        for unsafe in ("../private", "prediction-" + "0" * 32 + "/secret", "/etc/passwd", "x", None):
            with self.assertRaises(ValueError):
                self.store.get(unsafe)
        self.assertIsNone(self.store.get("prediction-" + "0" * 32))

    def test_escaped_symlink_is_not_followed(self):
        self.root.mkdir()
        prediction_id = "prediction-" + "1" * 32
        outside = self.root.parent / "private.json"
        outside.write_text(json.dumps({"id": prediction_id, "secret": True}))
        (self.root / (prediction_id + ".json")).symlink_to(outside)
        with self.assertRaises(ValueError):
            self.store.get(prediction_id)

    def test_invalid_json_payload_is_not_partially_saved(self):
        with self.assertRaises(ValueError):
            self.store.save({"result": float("nan")})
        self.assertFalse(self.root.exists())
        with self.assertRaises(ValueError):
            self.store.save([])

    def test_mismatched_stored_id_is_rejected(self):
        record = self.store.save({"value": 1})
        path = self.root / (record["id"] + ".json")
        path.write_text('{"id":"another-prediction"}')
        with self.assertRaises(ValueError):
            self.store.get(record["id"])


if __name__ == "__main__":
    unittest.main()
