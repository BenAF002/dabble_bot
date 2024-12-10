# Dabble Bot
This is an NLP deep learning project inspired by the board game Dictionary Dabble. Its objective is to generate plausible definitions for imaginary words. The model in `dabble_bot.py` is a decoder only transformer model with self-attention following the architecture of the decoder block presented in the paper "Attention is All You Need" (2017). The notebook `dabble_bot_gpt.ipynb` walks through the thought process underlying the model creation and training.

Sample model weights are stored in the `model_backup` folder. These weights were derived by training the model on a dataset of words derived from definitions in Stanford's WordNet lexical database with simplistic full-word tokenization. The model was trained on T4 GPUs via Google Colab.

Sample definitions may be drawn from the model as follows:

```python
# import model
import dabble_bot as dbb

# load model state
model = dbb.DabbleBot()
model.to(dbb.device)
model.load_state_dict(torch.load(path / 'model_backups/dabble_weights', weights_only=True))

# generate samples
model.generate(samples=10)
```
Note that you will need the curated data in the `curated_data` folder to generate samples from the model. This is because the model initializes encoding/decoding logic from these datasets, even when fine-tuning it on other datasets.

This is/was an educational exercise in deep learning, not functional coding, so it's not grade A code.
