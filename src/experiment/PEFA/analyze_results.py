#!/usr/bin/env python3
"""
Analyze results: elapsed time, token counts, and success rates
"""
import re
from pathlib import Path

def main():
    result_dir = Path("/home/hojoon/COHERENT/src/experiment/PEFA/result_log/gpt-5-mini")
    
    # 1. Calculate Elapsed Time from env files
    print("=" * 60)
    print("ENV FILES - ELAPSED TIME")
    print("=" * 60)
    
    total_elapsed = 0
    env_files = sorted([f for f in result_dir.glob("env*.txt") if f.stem.startswith("env") and f.stem[3:].isdigit()])
    
    for env_file in env_files:
        try:
            content = env_file.read_text()
            times = re.findall(r'Elapsed_Time: ([\d.]+)s', content)
            elapsed = sum(float(t) for t in times)
            print(f"{env_file.name:15s}: {elapsed:8.2f}s ({len(times):2d} tasks)")
            total_elapsed += elapsed
        except Exception as e:
            print(f"{env_file.name:15s}: Error - {e}")
    
    print("-" * 60)
    print(f"{'TOTAL':15s}: {total_elapsed:8.2f}s\n")
    
    # 2. Calculate Success Rates
    print("=" * 60)
    print("SUCCESS RATE ANALYSIS")
    print("=" * 60)
    
    steps_success = steps_fail = 0
    count_success = count_fail = 0
    
    env_stats = []
    
    for env_file in env_files:
        try:
            content = env_file.read_text()
            
            # Find all setps and count lines
            setps_matches = re.findall(r'setps: (\d+)/(\d+)', content)
            count_matches = re.findall(r'count: (\d+)/(\d+)', content)
            
            env_steps_success = env_steps_fail = 0
            env_count_success = env_count_fail = 0
            
            # Analyze steps
            for left, right in setps_matches:
                if int(left) > int(right):
                    steps_fail += 1
                    env_steps_fail += 1
                else:
                    steps_success += 1
                    env_steps_success += 1
            
            # Analyze count
            for left, right in count_matches:
                if int(left) > int(right):
                    count_fail += 1
                    env_count_fail += 1
                else:
                    count_success += 1
                    env_count_success += 1
            
            env_total_tasks = len(setps_matches)
            if env_total_tasks > 0:
                steps_rate = (env_steps_success / env_total_tasks) * 100
                count_rate = (env_count_success / env_total_tasks) * 100
            else:
                steps_rate = count_rate = 0
            
            env_stats.append({
                'name': env_file.name,
                'tasks': env_total_tasks,
                'steps_success': env_steps_success,
                'steps_fail': env_steps_fail,
                'steps_rate': steps_rate,
                'count_success': env_count_success,
                'count_fail': env_count_fail,
                'count_rate': count_rate
            })
            
        except Exception as e:
            print(f"{env_file.name}: Error - {e}")
    
    # Print per-env statistics
    print("\n[Per Environment]")
    print(f"{'File':<15} {'Tasks':<7} {'Steps Success':<15} {'Steps Rate':<12} {'Count Success':<15} {'Count Rate':<12}")
    print("-" * 85)
    for stat in env_stats:
        print(f"{stat['name']:<15} {stat['tasks']:<7} "
              f"{stat['steps_success']}/{stat['steps_fail']+stat['steps_success']:<13} "
              f"{stat['steps_rate']:>6.1f}%     "
              f"{stat['count_success']}/{stat['count_fail']+stat['count_success']:<13} "
              f"{stat['count_rate']:>6.1f}%")
    
    # Print overall statistics
    total_tasks = steps_success + steps_fail
    print("\n" + "=" * 60)
    print("[OVERALL SUCCESS RATES]")
    print("=" * 60)
    print(f"Total Tasks: {total_tasks}")
    print()
    print(f"✓ Steps-based Success Rate:")
    print(f"  - Success: {steps_success}/{total_tasks} = {(steps_success/total_tasks*100):.2f}%")
    print(f"  - Fail:    {steps_fail}/{total_tasks} = {(steps_fail/total_tasks*100):.2f}%")
    print()
    print(f"✓ Count-based Success Rate:")
    print(f"  - Success: {count_success}/{total_tasks} = {(count_success/total_tasks*100):.2f}%")
    print(f"  - Fail:    {count_fail}/{total_tasks} = {(count_fail/total_tasks*100):.2f}%")
    print()
    
    # 3. Calculate Token counts for GPT-5-MINI
    print("=" * 60)
    print("GPT-5-MINI TOKEN COUNTS")
    print("=" * 60)
    
    mini_input = mini_output = mini_total = 0
    mini_files = sorted(result_dir.glob("*gpt-5-mini_token.txt"))
    
    for token_file in mini_files:
        try:
            content = token_file.read_text()
            file_input = file_output = file_total = 0
            
            for line in content.strip().split('\n'):
                if line.strip():
                    match = re.search(r'INPUT: (\d+), OUTPUT: (\d+), TOTAL: (\d+)', line)
                    if match:
                        file_input += int(match.group(1))
                        file_output += int(match.group(2))
                        file_total += int(match.group(3))
            
            if file_total > 0:
                print(f"{token_file.name:50s}: {file_total:>10,}")
                mini_input += file_input
                mini_output += file_output
                mini_total += file_total
        except Exception as e:
            print(f"{token_file.name:50s}: Error - {e}")
    
    print("-" * 60)
    print(f"{'Total INPUT':50s}: {mini_input:>10,}")
    print(f"{'Total OUTPUT':50s}: {mini_output:>10,}")
    print(f"{'Total TOTAL':50s}: {mini_total:>10,}\n")
    
    # 4. Calculate Token counts for GPT-5-NANO
    print("=" * 60)
    print("GPT-5-NANO TOKEN COUNTS")
    print("=" * 60)
    
    nano_input = nano_output = nano_total = 0
    nano_files = sorted(result_dir.glob("*gpt-5-nano_token.txt"))
    
    for token_file in nano_files:
        try:
            content = token_file.read_text()
            file_input = file_output = file_total = 0
            
            for line in content.strip().split('\n'):
                if line.strip():
                    match = re.search(r'INPUT: (\d+), OUTPUT: (\d+), TOTAL: (\d+)', line)
                    if match:
                        file_input += int(match.group(1))
                        file_output += int(match.group(2))
                        file_total += int(match.group(3))
            
            if file_total > 0:
                print(f"{token_file.name:50s}: {file_total:>10,}")
                nano_input += file_input
                nano_output += file_output
                nano_total += file_total
        except Exception as e:
            print(f"{token_file.name:50s}: Error - {e}")
    
    print("-" * 60)
    print(f"{'Total INPUT':50s}: {nano_input:>10,}")
    print(f"{'Total OUTPUT':50s}: {nano_output:>10,}")
    print(f"{'Total TOTAL':50s}: {nano_total:>10,}\n")
    
    # 5. Final Summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Elapsed Time (env files):     {total_elapsed:>12.2f}s")
    print(f"Steps-based Success Rate:     {steps_success}/{total_tasks:>3} = {(steps_success/total_tasks*100):>6.2f}%")
    print(f"Count-based Success Rate:     {count_success}/{total_tasks:>3} = {(count_success/total_tasks*100):>6.2f}%")
    print(f"GPT-5-MINI Total Tokens:      {mini_total:>12,}")
    print(f"  - Input:                    {mini_input:>12,}")
    print(f"  - Output:                   {mini_output:>12,}")
    print(f"GPT-5-NANO Total Tokens:      {nano_total:>12,}")
    print(f"  - Input:                    {nano_input:>12,}")
    print(f"  - Output:                   {nano_output:>12,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
