import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualization script with command-line arguments")
    parser.add_argument("--date", choices=["all_day", "23.10.16_PM"], default="all_day", help="Date option")
    parser.add_argument("--altitude", choices=["15", "30", "80", "120", "all"], default="all", help="Altitude option")
    parser.add_argument("--eval_case", choices=["case1", "case2"], default="case1", help="Evaluation case")
    parser.add_argument("--sort_order", choices=["ascending", "descending"], default="ascending", help="Sort order")
    parser.add_argument("--num_vis", type=int, default=10, help="Number of visualizations")
    parser.add_argument("--rank_vis", type=int, default=10, help="Rank visualization")
    parser.add_argument("--custom_output_dir", type=str, default="./vis", help="Custom output directory for visual")
    parser.add_argument("--output_dir", type=str, default="logs/run", help="Output directory for logs")
    
    args = parser.parse_args()

    # Test case parameters
    date = args.date
    num_ids = "all"
    altitude = args.altitude
    eval_case = args.eval_case
    
    # Visualization parameters
    sort_order = args.sort_order
    num_vis = args.num_vis
    rank_vis = args.rank_vis
    
    output_dir = f"logs/{date}_{num_ids}_ids_{altitude}_meters"
    output_dir = args.output_dir
    dataset_subset = f"{date}_{num_ids}_ids_{altitude}_meters"
    test_weight = f"{output_dir}/best_model.pth.tar"
    custom_output_dir = args.custom_output_dir

    print(f"Eval case: {eval_case}, Sort order: {sort_order}, Altitude: {altitude}")
    eval_cmd = (
        f"python eval_all.py "
        f"--sort {sort_order} "
        f"--num_vis {num_vis} "
        f"--rank_vis {rank_vis} "
        f"--vis_output_dir {custom_output_dir} "
        f"DATASETS.SUBSET {dataset_subset} "
        f"DATASETS.SUBSUBSET {custom_output_dir} "
        f"DATASETS.EVAL_CASE {eval_case} "
        f"OUTPUT_DIR {output_dir} "
        f"TEST.WEIGHT {test_weight} "
    )
    
    os.system(eval_cmd)

if __name__ == "__main__":
    main()
