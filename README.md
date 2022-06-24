## SO WHAT IS THIS ANYWAY
It's a thingy that uses a neural net to guess what tone of voice you're using and then relays that to VRChat through OSC. Intended usage is to record your own training data so it's more accurate to your specific voice and speech mannerisms. Very pre-alpha, uploading so others can try it / contribute.

## VAGUE INSTRUCTIONS TO GET STARTED
1. Install Python 3.9.12 from the website
2. Set up a virtual environment
3. Activate virtual environment
4. `pip install -r requirements.txt`
5. Record yourself reading the script in a few different tones (I did neutral, excited, and sad)
6. Put all recordings for a given tone in a folder with the name you want, and put all those inside a folder called `dataset` or something.
7. Edit the config to point to that folder
8. `python .\emoflow.py train --config config.yaml`
9. Go grab a snack, this'll take a while
10. `python .\emoflow.py run --config config.yaml`
11. Look at the log output, and see how the confidence values for each category tend to be biased versus the tone you're actually using, and modify the `interpret_scores()` function to account for that
12. Set up your avatar to respond appropriately to the output