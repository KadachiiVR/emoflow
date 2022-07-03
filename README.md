## SO WHAT IS THIS ANYWAY
It's a thingy that uses a neural net to guess what tone of voice you're using and then relays that to VRChat through OSC. Intended usage is to record your own training data so it's more accurate to your specific voice and speech mannerisms. Very pre-alpha, uploading so others can try it / contribute.

## VAGUE INSTRUCTIONS TO GET STARTED
1. Install Python 3.9.12 from the website
2. Set up a virtual environment (for training)
3. Activate virtual environment
4. `pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111`
5. `pip install -r requirements.txt`
6. Set up ANOTHER virtual environment (for running)
7. Activate virtual environment
8. `pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu`
9. `pip install -r requirements.txt`
10. Switch back to the training environment
11. Record yourself reading the script in a few different tones (I did neutral, excited, and sad)
12. Put all recordings for a given tone in a folder with the name you want, and put all those inside a folder called `dataset` or something.
13. Edit the config to point to that folder
14. `python .\emoflow.py train --config config.yaml`
15. Go grab a snack, this'll take a few minutes
16. Switch to the running environment
17. `python .\emoflow.py run --config config.yaml`
18. Look at the log output, and see how the confidence values for each category tend to be biased versus the tone you're actually using, and modify the `interpret_scores()` function to account for that
19. Set up your avatar to respond appropriately to the output