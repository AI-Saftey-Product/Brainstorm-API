"""Llama adapter for making API calls to Llama models."""
import logging
from brainstorm.db.models.model import ModelDefinition
from brainstorm.core.adapters.base_adapter import BaseModelAdapter
import google.auth
import google.auth.transport.requests
import requests

logger = logging.getLogger(__name__)


class GCP_MAAS_NLPAdapter():
    """Adapter for Llama NLP models."""
    
    def __init__(self, config: ModelDefinition):
        """
        Initialize the Llama adapter.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__()

        self.creds, self.project = google.auth.default()

        # creds.valid is False, and creds.token is None
        # Need to refresh credentials to populate those
        if self.creds.valid == False:
            auth_req = google.auth.transport.requests.Request()
            self.creds.refresh(auth_req)

        self.config = config

    async def generate(self, prompt: str, **parameters) -> str:
        ENDPOINT = "us-east5-aiplatform.googleapis.com"
        REGION = "us-east5"

        headers = {
            "Authorization": f"Bearer {self.creds.token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "meta/llama-4-maverick-17b-128e-instruct-maas",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        res = requests.post(
            f"https://{ENDPOINT}/v1/projects/{self.project}/locations/{REGION}/endpoints/openapi/chat/completions",
            json=payload,
            headers=headers
        )

        message = res.json()['choices'][0]['message']['content']
        return message
