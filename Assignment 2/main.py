import os
from src.experiment_runner import run_experiment
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings if needed
warnings.filterwarnings("ignore", category=RuntimeWarning) 



def main():
    """
    Main function to run the experiment with a specified dataset.
    """
    os.environ['NUMEXPR_MAX_THREADS'] = '12'

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use a path relative to the script location (in the same directory)
    root_dir = os.path.join(script_dir, 'op_spam_v1.4', 'op_spam_v1.4')
    
    # Run the experiment
    print("Starting experiment...")  
    print(f"Using dataset at: {root_dir}")
    results = run_experiment(root_dir)
   
    
    
    # Display the final results
    print("\nFinal Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

if __name__ == "__main__":
    main()
