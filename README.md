## SO WHAT IS THIS ANYWAY
It's a thingy that uses a neural net to guess what tone of voice you're using and then relays that to VRChat through OSC. Intended usage is to record your own training data so it's more accurate to your specific voice and speech mannerisms. Very pre-alpha, uploading so others can try it / contribute.

## VAGUE INSTRUCTIONS TO GET STARTED
1. Install Python 3.9.12 from the website
2. Set up a virtual environment
3. Activate virtual environment
4. `pip install -r requirements.txt`
5. Edit the config I guess
6. `python .\emoflow.py train --config config.yaml`