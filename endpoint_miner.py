import os
import time
import argparse
import bittensor
import requests
import json
import traceback
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

class EndpointMiner:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--endpoint.verify_token', type=str, help='Auth')
        parser.add_argument('--endpoint.url', type=str, help='Endpoint')

    def config(self) -> "bittensor.Config":
        parser = argparse.ArgumentParser(description="Bittensor Miner Configs")
        self.add_args(parser)
        return bittensor.config(parser)

    def __init__(self, *args, **kwargs):
        super(EndpointMiner, self).__init__(*args, **kwargs)
        print("Initialized")
        
        # Initialize HuggingFace model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

    def prompt(self, synapse: dict) -> str:
        start_time = time.time()
        errored = False
        generation = ""
        error_msg = "None"
        
        try:
            # Prepare input for model
            messages = [
                {"role": role, "content": message}
                for role, message in zip(synapse["roles"], synapse["messages"])
            ]
            
            print("Messages:", messages)
            input_text = ' '.join([message["content"] for message in messages])
            
            # Generate response using Hugging Face model
            inputs = self.tokenizer(input_text, return_tensors="pt")
            output = self.model.generate(inputs["input_ids"], max_length=200)
            generation = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        except Exception as e:
            errored = True
            traceback.print_exc()
            error_msg = str(e)

        # Delay before responding
        time_to_sleep = 9.6 - (time.time() - start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        
        print("Generated:", generation)
        return generation

if __name__ == "__main__":
    print("Starting")
    miner = EndpointMiner()
    while True:
        print("Running miner...", time.time())
        time.sleep(10)

