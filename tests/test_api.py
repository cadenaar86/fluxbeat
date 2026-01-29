import unittest
from fastapi.testclient import TestClient
import sys
import os
import io
import numpy as np
import soundfile as sf

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.server import app

class TestAPI(unittest.TestCase):
    def test_health(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['status'], 'ok')

    def test_predict_endpoint(self):
        with TestClient(app) as client:
            # Create a dummy small wav file
            dummy_audio = np.random.uniform(-1, 1, size=(22050*1))
            # Save to buffer
            buf = io.BytesIO()
            sf.write(buf, dummy_audio, 22050, format='WAV')
            buf.seek(0)
            
            response = client.post(
                "/predict",
                files={"file": ("test.wav", buf, "audio/wav")}
            )
            
            if response.status_code != 200:
                print(response.json())
                
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("predictions", data)
            self.assertIn("top_genre", data)

if __name__ == '__main__':
    unittest.main()
