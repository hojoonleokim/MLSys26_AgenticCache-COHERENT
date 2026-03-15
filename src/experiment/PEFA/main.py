import json
import time
import subprocess
from LLM_agent import LLM_agent
from args import get_args
from LLM_oracle import ArenaMP
from get_env_info import Get_env_info
import traceback
import argparse
import os

args = get_args()

# Determine branch name for result organization
branch_name = args.branch
if not branch_name:
    try:
        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        branch_name = 'unknown'

# Setup result directory: results/<branch>/<lm_id>/<env>/
args.branch = branch_name
result_dir = f'./results/{branch_name}/{args.lm_id}/{args.env}'
os.makedirs(result_dir, exist_ok=True)

def write_log_to_file(log_message, file_name=None):
        if file_name is None:
            file_name = os.path.join(result_dir, 'run_log.txt')
        with open(file_name, 'a') as file:
            file.write(log_message + '\n')


if __name__ == '__main__':

    
    with open(f'./env/{args.env}.json') as file:
        data = json.load(file)
    '''
    The task uesed in our paper is as follows:
    Env0: [2, 4, 9, 10, 11, 15, 16, 20]
    Env1: [1, 3, 7, 8, 11, 10, 16, 20]
    Env2: [3, 5, 6, 7, 10, 11, 16 ,17]
    Env3: [2, 4, 6, 7, 10, 16, 17, 19]
    Env4: [0, 1, 7, 10, 12, 17, 18, 19]
    '''
    tasklist = args.task
    steps_list, tasks_result = [], []
    success_tasks, failed_tasks = [], []
    episode_results = {}
    episode_times = {}

    for task_id in tasklist:
        # Create episode_dir for per-episode structured data
        episode_dir = os.path.join(result_dir, f'episode_{task_id}')
        os.makedirs(episode_dir, exist_ok=True)

        env_id = data[task_id]["env_id"]
        task_name = data[task_id]["task_name"]
        graph = data[task_id]["init_graph"]
        task_goal = data[task_id]["task_goal"]
        goal_instruction = data[task_id]["goal_instruction"]
        ground_truth_step_num = data[task_id]["ground_truth_step_num"]

        test_results = {}
        agent = []
        for node in data[task_id]["init_graph"]["nodes"]:
            if node["category"] == "Agents":
                agent.append([node["class_name"],node["id"]])

        agent_nodes = []
        for a in agent:
            agent_node = [node for node in graph['nodes'] if node['id'] == a[1]]
            agent_nodes += agent_node

        dict_list = []
        for i, agent_name in enumerate(agent):
            dict_item = {
                'agent_id': i,
                'args': args,
                'agent_node': agent_nodes[i],
                'init_graph': graph,
            }
            dict_list.append(dict_item)


        def env_fn():
            return Get_env_info(
                                task_id = task_id,
                                env_id = env_id,
                                task_name = task_name,
                                graph = graph,
                                task_goal = task_goal,
                                goal_instruction = goal_instruction,
                                ground_truth_step_num = ground_truth_step_num,
                                agent = agent,
                                num_agent = len(agent)
                                )

        def LLM_agent_fn(args_llm):

            return LLM_agent(**args_llm)

        agents = [LLM_agent_fn(dict_list[i]) for i in range(len(dict_list))]
        # print(len(agents))
        # input('s')
        arena = ArenaMP(env_fn, agents, args)

        steps = 0

        task_start_time = time.time()
        try:
            success, steps, saved_info = arena.run() # run the environment
        except Exception as e:

            print(f"An error occurred: {e}")
            traceback.print_exc()
            error_info = traceback.format_exc()
            write_log_to_file(f"An error occurred: {e}")
            write_log_to_file(error_info+'\n\n')
            success = False
            saved_info = []
            raise Exception
        task_elapsed = time.time() - task_start_time
        print(f'---------Env:{env_id}---Task:{task_id}-------------------------')
        print('success' if success else 'failure')
        print('steps:', steps)
        print('-------------------------------------')
        steps_list.append(steps)
        tasks_result.append(1 if success else 0)
        if success:
            success_tasks.append(task_id)
        else:
            failed_tasks.append(task_id)

        # Save episode result
        episode_results[str(task_id)] = {
            "task_name": task_name,
            "success": success,
            "steps": steps,
            "oracle_count": arena.oracle_count,
            "gt_steps": ground_truth_step_num,
            "time": task_elapsed
        }

        # Save per-episode detailed data (episode_dir already created above)

        # Save step-by-step timeline (exclude env_graph to save space)
        timeline = []
        for si in saved_info:
            entry = {k: v for k, v in si.items() if k != 'env_graph'}
            timeline.append(entry)
        with open(os.path.join(episode_dir, 'timeline.jsonl'), 'w') as f:
            for entry in timeline:
                f.write(json.dumps(entry, default=str) + '\n')

        # Save per-agent action history
        for agent_obj in agents:
            agent_class = agent_obj.agent_node['class_name'].replace(' ', '_')
            agent_real_id = agent_obj.agent_node['id']
            agent_file = os.path.join(episode_dir, f'agent_{agent_class}_{agent_real_id}.jsonl')
            with open(agent_file, 'w') as f:
                for step_num, action in sorted(agent_obj.action_history.items()):
                    entry = {
                        "step": step_num,
                        "action": action if action is not None else "[wait]"
                    }
                    f.write(json.dumps(entry) + '\n')
        
        # Cleanup: Wait for all async queries to complete and close pools before next episode
        print("Cleaning up resources before next episode...")
        
        # Step 1: Wait for all pending async oracle queries to complete
        if hasattr(arena, 'async_results'):
            print("Waiting for pending Oracle queries...")
            for i in range(len(arena.async_results)):
                if arena.async_results[i] is not None:
                    try:
                        arena.async_results[i].get(timeout=30)  # Wait up to 30 seconds
                    except Exception as e:
                        print(f"Warning: Oracle async query {i} failed: {e}")
                    arena.async_results[i] = None
        
        # Step 2: Wait for all pending async agent queries to complete
        for idx, agent in enumerate(agents):
            if hasattr(agent, 'async_results'):
                print(f"Waiting for pending Agent {idx} queries...")
                for i in range(len(agent.async_results)):
                    if agent.async_results[i] is not None:
                        try:
                            agent.async_results[i].get(timeout=30)
                        except Exception as e:
                            print(f"Warning: Agent {idx} async query {i} failed: {e}")
                        agent.async_results[i] = None
        
        # Step 3: Close and join pools (manager is still needed during join)
        if hasattr(arena, 'oracle_pool'):
            print("Closing Oracle pool...")
            arena.oracle_pool.close()
            arena.oracle_pool.join()
        
        for idx, agent in enumerate(agents):
            if hasattr(agent, 'agent_pool'):
                print(f"Closing Agent {idx} pool...")
                agent.agent_pool.close()
                agent.agent_pool.join()
        
        # Step 4: Shutdown managers AFTER pools are completely done
        if hasattr(arena, 'manager'):
            arena.manager.shutdown()
        
        for idx, agent in enumerate(agents):
            if hasattr(agent, 'manager'):
                agent.manager.shutdown()
        
        print("Resource cleanup complete. Ready for next episode.\n")

    # Save aggregated eval_result.json
    avg_steps = sum(steps_list)/len(steps_list) if len(steps_list) > 0 else None
    succ_steps = [steps_list[i] for i in range(len(tasks_result)) if tasks_result[i] == 1]
    eval_result = {
        "branch": branch_name,
        "model": args.lm_id,
        "env": args.env,
        "avg_succ": sum(tasks_result) / len(tasks_result) if tasks_result else 0.0,
        "avg_succ_steps": sum(succ_steps) / len(succ_steps) if succ_steps else None,
        "episode_results": episode_results
    }
    with open(os.path.join(result_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f, indent=4)
    print(f'[Results saved to {result_dir}/eval_result.json]')

    write_log_to_file(f'average steps: {avg_steps}')
    write_log_to_file(f'successful tasks: {success_tasks if len(success_tasks) > 0 else None}')
    write_log_to_file(f'failed tasks: {failed_tasks if len(failed_tasks) > 0 else None}')
    print('average steps:', avg_steps)
    print('successful tasks:', success_tasks if len(success_tasks) > 0 else None )
    print('failed tasks:', failed_tasks if len(failed_tasks) > 0 else None)    
