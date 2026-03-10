import os, time
import importlib

def banner(text):
    print("\n" + "█" * 64)
    print(f"█  {text:^58}  █")
    print("█" * 64 + "\n")


def run_task(module_name, label):
    banner(label)
    t0 = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    print(f"\n⏱  {label} completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_task("src.resampling",       "Task 1: Resampling Methods")
    run_task("src.subset_selection", "Task 2a: Subset Selection")
    run_task("src.shrinkage",       "Task 2b: Shrinkage (Ridge & Lasso)")
    run_task("src.pca_pls",         "Task 2c: PCA & PLS")

    print("\n✅ All tasks completed!")