import pickle

from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("AI4Math/MathVista")

filename = 'model_eval_1_8b_finetuned.pkl'
with open(filename, 'rb') as f:
  eval_1_8 = pickle.load(f)
  
print(eval_1_8)
  


