import sys
sys.path.insert(0, '.')
from config import Config, ExperimentMode
from train import train

cfg = Config(
    experiment_name='smoke_test',
    mode=ExperimentMode.A,
    num_epochs=2,
    batch_size=256,
    num_workers=0,
    pin_memory=False,
    early_stopping=False,
    results_dir='./results/smoke',
    data_dir='./data',
    seed=42,
)
model, metrics = train(cfg)
val_acc = metrics["final_val_accuracy"]
n_params = metrics["total_parameters"]
print("val_acc=%.2f%%" % (val_acc * 100))
print("params=%d" % n_params)
print("=== End-to-end pipeline: PASS ===")
