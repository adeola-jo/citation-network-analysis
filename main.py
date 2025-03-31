import os
import argparse
import subprocess
import time

def print_header(title):
    """Print a formatted header for better readability."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def run_script(script_name):
    """Run a Python script and capture the output."""
    print_header(f"Running {script_name}")
    start_time = time.time()
    
    try:
        # Run the script and capture the output
        result = subprocess.run(['python', script_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        
        # Print the output
        print(result.stdout)
        
        # Print any errors
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        elapsed_time = time.time() - start_time
        print(f"\nScript completed in {elapsed_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Citation Network Analysis")
    parser.add_argument('--basic', action='store_true', help='Run basic GNN model only')
    parser.add_argument('--compare', action='store_true', help='Run architecture comparison')
    parser.add_argument('--network', action='store_true', help='Run network analysis')
    parser.add_argument('--link', action='store_true', help='Run link prediction')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    
    args = parser.parse_args()
    
    # If no specific flag is provided, run the basic analysis
    if not (args.basic or args.compare or args.network or args.link or args.all):
        args.basic = True
    
    # Run the requested analyses
    if args.basic or args.all:
        run_script('citation_network_gnn.py')
    
    if args.compare or args.all:
        run_script('compare_gnn_architectures.py')
    
    if args.network or args.all:
        run_script('network_analysis.py')
    
    if args.link or args.all:
        run_script('link_prediction.py')
    
    print_header("Citation Network Analysis Complete")
    print("Results and visualizations have been saved to the output directory.")
    print("\nYou can find the following files:")
    
    # List the output files
    output_files = [f for f in os.listdir('.') if f.endswith(('.png', '.csv', '.json'))]
    for file in sorted(output_files):
        print(f" - {file}")

if __name__ == "__main__":
    main()
