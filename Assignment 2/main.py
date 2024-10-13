from src.experiment_runner import run_experiment

def main():
    """
    Main function to run the experiment with a specified dataset.
    """
    # Specify the root directory of the dataset
    root_dir = 'op_spam_v1.4/'
    
    # Run the experiment
    print("Starting experiment...")
    results = run_experiment(root_dir)
    
    # Display the final results
    print("\nFinal Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

if __name__ == "__main__":
    main()
