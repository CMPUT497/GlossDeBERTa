import subprocess
import sys
import os

# --- Configuration ---
# You can adjust these if you run into memory errors
BATCH_SIZE = "32" 
GRADIENT_ACCUMULATION = "1"
EPOCHS = "4.0"
LAYERS = "7"  # Truncation for speed
LR = "2e-5"

# Paths
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "gloss_deberta_full_7layers")
TRAIN_FILE = os.path.join(BASE_DIR, "Training_Corpora", "SemCor", "semcor_train_token_cls.csv")
TEST_FILE = os.path.join(BASE_DIR, "Evaluation_Datasets", "semeval2007", "semeval2007_test_token_cls.csv")
SENSEVAL3_FILE = os.path.join(BASE_DIR, "Evaluation_Datasets", "senseval3", "senseval3_test_token_cls.csv")
WORDNET_DIR = os.path.join(BASE_DIR, "wordnet")

def run_command(command, step_name):
    """Runs a shell command and prints status."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"Running: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        # shell=False is safer; sys.executable ensures we use the same python env
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred during {step_name}. Exit code: {e.returncode}")
        sys.exit(1)

def main():
    # --- 1. Run Preparation Script ---
    # This generates the CSV files and pickle dictionaries from raw data
    if not os.path.exists("./wordnet/lemma2index_dict.pkl"):
        run_command(["bash", "preparation.sh"], "Data Preparation")
    else:
        print("Skipping preparation (files already exist).")

    # --- 2. Run Full Training ---
    train_cmd = [
        sys.executable, "run_classifier_WSD_token.py",
        "--task_name", "WSD",
        "--train_data_dir", TRAIN_FILE,
        "--eval_data_dir", TEST_FILE,
        "--label_data_dir", WORDNET_DIR,
        "--output_dir", OUTPUT_DIR,
        "--bert_model", "microsoft/deberta-v3-base",
        "--do_train",
        "--do_eval",
        "--max_seq_length", "128",
        "--train_batch_size", BATCH_SIZE,
        "--learning_rate", LR,
        "--num_train_epochs", EPOCHS,
        "--seed", "42",
        "--gradient_accumulation_steps", GRADIENT_ACCUMULATION,
        "--num_layers", LAYERS
    ]
    run_command(train_cmd, "Training (GlossDeBERTa)")

    # --- 3. Run Evaluation (Senseval 3) ---
    # We point to epoch 4 (or whatever EPOCHS is set to)
    final_model_path = os.path.join(OUTPUT_DIR, str(int(float(EPOCHS))))
    test_output_dir = os.path.join(OUTPUT_DIR, "test_senseval3")

    eval_cmd = [
        sys.executable, "run_classifier_WSD_token.py",
        "--task_name", "WSD",
        "--eval_data_dir", SENSEVAL3_FILE,
        "--label_data_dir", WORDNET_DIR,
        "--output_dir", test_output_dir,
        "--bert_model", final_model_path,
        "--do_test",
        "--max_seq_length", "128",
        "--eval_batch_size", "32",
        "--num_layers", LAYERS
    ]
    run_command(eval_cmd, "Evaluation (Senseval 3)")

    print(f"\nPipeline finished successfully! Results saved in: {test_output_dir}")

if __name__ == "__main__":
    main()