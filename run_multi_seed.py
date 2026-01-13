"""
Run ASG-SI with multiple seeds to generate error bars and learning curves
"""
import json
import sys
sys.path.insert(0, '/home/kengpu/ASG-SI')

from asg_si_demo import ASGSISystem

def run_experiment(seed, iters=6, num_tasks=80):
    """Run single experiment with given seed"""
    system = ASGSISystem(seed=seed, audit_dir=f"audit_logs_seed_{seed}", memory_capacity=8, use_model=False)
    summaries = system.train(iters=iters, num_tasks=num_tasks)
    return summaries

def main():
    seeds = [7, 42, 123, 456, 789]
    all_results = {}
    num_tasks = 80
    iters = 6

    print("Running ASG-SI with multiple seeds for error bars...")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    for seed in seeds:
        print(f"\nRunning seed {seed}...")
        summaries = run_experiment(seed=seed, iters=iters, num_tasks=num_tasks)
        all_results[seed] = summaries

        # Print summary for this seed
        print(f"Seed {seed} - Final results:")
        final = summaries[-1]
        print(f"  Success rate: {final['success_rate']:.3f}")
        print(f"  Avg reward: {final['avg_reward']:.3f}")
        print(f"  Skill reuse: {final['actions']['skill']}/{num_tasks} ({final['actions']['skill']/num_tasks*100:.1f}%)")

    # Save results
    with open('multi_seed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("Results saved to multi_seed_results.json")

    # Compute statistics across seeds
    import statistics

    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS ACROSS SEEDS")
    print("=" * 80)

    for iter_num in range(6):
        success_rates = [all_results[seed][iter_num]['success_rate'] for seed in seeds]
        rewards = [all_results[seed][iter_num]['avg_reward'] for seed in seeds]
        skill_uses = [all_results[seed][iter_num]['actions']['skill'] for seed in seeds]

        print(f"\nIteration {iter_num}:")
        print(f"  Success rate: {statistics.mean(success_rates):.3f} ± {statistics.stdev(success_rates) if len(success_rates) > 1 else 0:.3f}")
        print(f"  Avg reward: {statistics.mean(rewards):.3f} ± {statistics.stdev(rewards) if len(rewards) > 1 else 0:.3f}")
        print(f"  Skill actions: {statistics.mean(skill_uses):.1f} ± {statistics.stdev(skill_uses) if len(skill_uses) > 1 else 0:.1f}")

if __name__ == "__main__":
    main()
