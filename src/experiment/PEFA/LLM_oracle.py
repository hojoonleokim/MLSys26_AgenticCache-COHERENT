
import copy
import numpy as np
from tqdm import tqdm
import time
import os
import json
from openai import OpenAIError,OpenAI
import backoff
import traceback
from multiprocessing import Pool, Manager
import torch

# Global variables for the worker process
_oracle_worker_client = None
_oracle_worker_lm_id = None
_oracle_worker_debug = None

def write_token_log_to_file( log_message=None, file_name=None):
    file_name = os.path.join(_oracle_worker_result_dir, f'oracle_{_oracle_worker_lm_id}_token.txt')
    with open(file_name, 'a') as file:  
        file.write(log_message + '\n')

def write_oracle_log_to_file( log_message=None, file_name=None):
    file_name = os.path.join(_oracle_worker_result_dir, f'oracle_{_oracle_worker_lm_id}_log.txt')
    with open(file_name, 'a') as file:  
        file.write(log_message + '\n')

def _pool_init_oracle(oracle_init_args):
	"""Initialize the OpenAI client in the worker process for oracle"""
	global _oracle_worker_client, _oracle_worker_lm_id, _oracle_worker_debug, _oracle_worker_result_dir
	
	api_key = oracle_init_args['api_key']
	organization = oracle_init_args['organization']
	_oracle_worker_lm_id = oracle_init_args['lm_id']
	_oracle_worker_debug = oracle_init_args['debug']
	_oracle_worker_result_dir = oracle_init_args['result_dir']
	os.makedirs(_oracle_worker_result_dir, exist_ok=True)
	
	_oracle_worker_client = OpenAI(api_key=api_key, organization=organization)

def _pool_check_init_oracle():
	"""Dummy function to check if pool initialization is complete"""
	global _oracle_worker_client
	return _oracle_worker_client is not None

def _oracle_worker_generate_single(oracle_prompt_path, obs2text, goal_instruction, num_agent, dialogue_history, action_history, sampling_params, done_flag, result_dict):
	"""Worker function to generate oracle response in a single query"""
	global _oracle_worker_client, _oracle_worker_lm_id, _oracle_worker_debug
	
	total_usage = 0
	try:
		# Build oracle prompt in worker process
		with open(oracle_prompt_path, 'r') as f:
			oracle_prompt = f.read()
		oracle_prompt = oracle_prompt.replace('#AGENT_OBSERVATIONS#', obs2text)
		oracle_prompt = oracle_prompt.replace('#TASK_GOAL#', goal_instruction)
		oracle_prompt = oracle_prompt.replace('#NUMBER_AGENTS#', str(num_agent))
		oracle_prompt = oracle_prompt.replace('#DIALOGUE_HISTORY#', dialogue_history)
		oracle_prompt = oracle_prompt.replace('#ACTION_HISTORY#', action_history)
		# Single generation: Oracle response in correct format
		prompt_copy = [{"role": "system", "content": "You are a helper assistant."}, 
		               {"role": "user", "content": oracle_prompt}]
		write_oracle_log_to_file(f"Oracle Prompt: {oracle_prompt}")
		response = _oracle_worker_client.chat.completions.create(
			model=_oracle_worker_lm_id,
			messages=prompt_copy,
			**sampling_params
		)
		# Extract and log token usage
		prompt_tokens = response.usage.prompt_tokens
		completion_tokens = response.usage.completion_tokens
		total_tokens = response.usage.total_tokens
		write_token_log_to_file(f"INPUT: {prompt_tokens}, OUTPUT: {completion_tokens}, TOTAL: {total_tokens}")

		if _oracle_worker_debug:
			os.makedirs('./log', exist_ok=True)
			with open(f"./chat_raw.json", 'a') as f:
				f.write(json.dumps(response.model_dump(), indent=4))
				f.write('\n')
		
		oracle_message = response.choices[0].message.content
		
		if 'gpt-4-0125-preview' in _oracle_worker_lm_id:
			total_usage += response.usage.prompt_tokens * 0.01 / 1000 + response.usage.completion_tokens * 0.03 / 1000
		elif 'gpt-3.5-turbo-1106' in _oracle_worker_lm_id:
			total_usage += response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
		
		# Store results in shared dict (oracle_message is now the final formatted message)
		result_dict['oracle_message'] = oracle_message
		result_dict['usage'] = total_usage
		done_flag.value = True
		write_oracle_log_to_file(f"Oracle Response: {oracle_message}")
		return oracle_message, total_usage
	except OpenAIError as e:
		print(e)
		result_dict['error'] = str(e)
		done_flag.value = True
		raise e

# @ray.remote
class ArenaMP(object):
    def __init__(self, environment_fn, agent_fn, args , run_predefined_actions=False):
        # run_predefined_actions is a parameter that you can use predefined_actions.json to strictly set the agents' actions instead of using algorithm to calculate the action.
        
        self.env_fn = environment_fn
        self.agents = agent_fn
        self.args = args
        self.num_agents = len(agent_fn)
        self.task_goal = None
        self.debug = args.debug
        print("Init Env")
        self.env = environment_fn()
        self.run_predefined_actions = run_predefined_actions

        self.oracle_prompt_path = args.oracle_prompt_path
        self.quadrotor_prompt_path = args.quadrotor_prompt_path
        self.robot_dog_prompt_path = args.robot_dog_prompt_path
        self.robot_arm_prompt_path = args.robot_arm_prompt_path

        self.dialogue_history = ""
        self.total_dialogue_history = []  # list of (step, text) tuples
        self.chat = True
        self.source = args.source
        self.lm_id = args.lm_id
        # self.lm_id = 'gpt-3.5-turbo-1106'
        self.device = None
        self.sampling_parameters = None
        self.total_cost = 0

        self.last_done = False
        self.last_task_results = None
        self.last_satisfied = None
        self.last_unsatisfied = None
        self.oracle_count = 0
        self.last_instructed_agent_id = None
        
        # Logging: results/<branch>/<lm_id>/<env>/
        self.result_dir = f'./results/{args.branch}/{self.lm_id}/{args.env}'
        os.makedirs(self.result_dir, exist_ok=True)
        self.record_dir = os.path.join(self.result_dir, f'{args.env}.txt')
        self.action_record_dir = os.path.join(self.result_dir, f'{args.env}_action.txt')

        if self.source == 'openai':
            api_key = args.api_key # your openai api key
            organization = args.organization # your openai organization

            if self.chat:
                # GPT-5 models require different parameters
                if 'gpt-5' in self.lm_id:
                    self.sampling_params = {
                        # "max_completion_tokens": args.max_tokens,
                        # "temperature": args.t,  # GPT-5 only supports default temperature (1)
                        # "top_p": 1.0,
                        "n": args.n
                    }
                else:
                    self.sampling_params = {
                        "max_tokens": args.max_tokens,
                        "temperature": args.t,
                        # "top_p": args.top_p,
                        "n": args.n
                    }
            
            # Create process pool with initialization
            oracle_init_args = {
                'api_key': api_key,
                'organization': organization,
                'lm_id': self.lm_id,
                'debug': self.debug,
                'result_dir': self.result_dir
            }
            
            self.oracle_pool = Pool(
                processes=1,
                initializer=_pool_init_oracle,
                initargs=(oracle_init_args,)
            )
            
            # Wait for pool initialization to complete
            self.oracle_pool.apply(_pool_check_init_oracle)
            
            self.manager = Manager()
            self.use_pool = True
            self.async_result = None
            self.oracle_done_flag = self.manager.Value('b', False)
            self.oracle_result_dict = self.manager.dict()
        else:
            raise ValueError("Only 'openai' source is currently supported")
    
    def submit_async_oracle_query(self, oracle_prompt_path, obs2text, goal_instruction, num_agent, dialogue_history, action_history, sampling_params):
        """Submit async oracle query (single API call)
        
        Args:
            oracle_prompt_path: Path to oracle prompt template file
            obs2text: Observation text
            goal_instruction: Task goal instruction
            num_agent: Number of agents
            dialogue_history: Dialogue history string
            action_history: Action history string
            sampling_params: Sampling parameters
        """
        if not self.use_pool:
            raise ValueError("Pool not initialized")
        
        if self.async_result is not None:
            print("Oracle async query already in progress")
            return
        
        # Reset shared memory for new query
        self.oracle_done_flag.value = False
        self.oracle_result_dict.clear()
        self.async_result = self.oracle_pool.apply_async(
            _oracle_worker_generate_single,
            (oracle_prompt_path, obs2text, goal_instruction, num_agent, dialogue_history, action_history, sampling_params, self.oracle_done_flag, self.oracle_result_dict)
        )
    
    def check_async_oracle_result(self):
        """Check if async oracle query is complete and return results"""
        if self.async_result is not None and self.async_result.ready():
            # Get the result (this should be immediate since ready() is True)
            self.async_result.get(timeout=10)
            oracle_message = self.oracle_result_dict.get('oracle_message', None)
            usage = self.oracle_result_dict.get('usage', 0)
            
            # Clear the async result
            self.async_result = None
            self.oracle_done_flag.value = False
            self.oracle_count += 1
            return oracle_message
        
        return None
    
    def __del__(self):
        """Cleanup the process pool when the object is destroyed"""
        if hasattr(self, 'oracle_pool'):
            self.oracle_pool.close()
            self.oracle_pool.join()
        if hasattr(self, 'manager'):
            self.manager.shutdown()

    def _format_dialogue(self, window=10):
        """Format recent dialogue history with step-based numbering.
        Same step → same base number; multiple entries per step get sub-indices.
        """
        entries = self.total_dialogue_history[-window:]
        if not entries:
            return ""
        from collections import Counter
        step_counts = Counter(s for s, _ in entries)
        step_seen = Counter()
        formatted = []
        for step, text in entries:
            step_seen[step] += 1
            if step_counts[step] == 1:
                formatted.append(f"[{step}]、{text}")
            else:
                formatted.append(f"[{step}.{step_seen[step]}]、{text}")
        return '\n'.join(formatted)

    def agent_obs2text(self, observation, agent_id):
        text = ""
        observation = observation[agent_id]
       
        id2node = {node['id']: node for node in observation["nodes"]}
        agent_class = id2node[int(self.env.id_name_dict[agent_id][1])]["class_name"]
        with_quadrotor_id = None
        for node in observation["nodes"]:
            if node["category"] == "Agents" and self.env.id_name_dict[agent_id][1] == node["id"]:
                # agent_node = node
                text += "I am <" + node["class_name"] +">(" + str(node["id"]) + "). "   
                if len(node['states']) != 0:
                    states = ', '.join(node['states'])
                    text += "Now my state is: " + states + ". "
                for edge in observation["edges"]:
                    if edge["from_id"] == node["id"]:
                        text += "I am " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). "
                    if edge["relation_type"] == "WITH":
                        with_quadrotor_id = edge["to_id"]
                text += '\n'


        for node in observation["nodes"]:
            if node["category"] == "Rooms" and node["id"] == observation["agent_in_room_id"]:
                text += "Now I am in the <"+ node["class_name"] +">(" + str(node["id"]) + "). In this room, I can see : \n"
        for node in observation["nodes"]:
            if node["id"] != self.env.id_name_dict[agent_id][1] and node["category"] != "Rooms":
                text += "<" + node["class_name"] +">(" + str(node["id"]) + "). "
                if len(node['properties']) != 0:
                    properties = ', '.join(node['properties'])
                    text += "Its properties are: " + properties + ". "
                if len(node['states']) !=0 :
                    states = ', '.join(node['states'])
                    text += "Now its state is: " + states + ". \n"
                else:
                    text += '\n'
        text += "These objects have a certain position relationship with each other: \n"
        for node in observation["nodes"]:
            if node["id"] != self.env.id_name_dict[agent_id][1] and node["category"] != "Rooms":
                for edge in observation["edges"]:
                    if edge["from_id"] == node["id"]: 
                    # if edge["from_id"] == node["id"] and edge["relation_type"] != "WITH" : #WITH is exclusive to quadrotor and basket
                        if edge["from_id"] == with_quadrotor_id and agent_class == 'quadrotor':
                            text += "The <" + node["class_name"] +">(" + str(node["id"]) + ") is with me LAND " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        elif edge["relation_type"] == "LEADING TO":
                            text += "The <" + node["class_name"] +">(" + str(node["id"]) + ") is " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        else:
                            text += "The <" + node["class_name"] +">(" + str(node["id"]) + ") is " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
        for edge in observation["edges"]:
            if edge["relation_type"] == "WITH" and agent_class == 'quadrotor':
                in_basket = False
                text += "I have a <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + ") with me. " 
                for edges in observation["edges"]:
                    if edges["to_id"] == edge["to_id"] and edges["relation_type"] == "INSIDE" :
                        text += "<" + id2node[edges["from_id"]]["class_name"] + ">("+ str(edges["from_id"]) + ") is in my <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        in_basket = True    
                if in_basket == False:
                    text += "But nothing is in my <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
            if edge["relation_type"] == "HOLD" and agent_class != 'quadrotor':
                text += "I am holding a <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + ") in my hand. \n"    
        # print(text)
        return text
    
    def write_log_to_file(self,log_message, file_name = None):
        file_name = self.record_dir
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')  

    def write_action_to_file(self, log_message=None, file_name=None):
        file_name = self.action_record_dir
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')

    def step(self):
        if self.env.steps == 0:
            pass

        obs = self.env.get_observations()
        id_name_dict = self.env.id_name_dict

        obs2text = ''
        for i in range(self.num_agents):
            obs2text += self.agent_obs2text(obs, i) + '\n'

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print((f"@@@@@@@@@@@@@@@@@@@@@@@@ Task_ID: {self.env.task_id} @@@@@@@@@@@"))
        print(f"$$$$$$$$$$$$$$$$$$$$$$$ Step:{self.env.steps} $$$$$$$$$$$$$$$$$$$$$$$")
        print(self.env.goal_instruction)
        
        # Check if Oracle result is ready
        oracle_message = self.check_async_oracle_result()
        
        if oracle_message is not None:
            # Oracle result is ready, use it and immediately submit next query
            print(f"Oracle result received: {oracle_message}")
            
            # Collect action history from all agents (one line per agent)
            agent_action_lines = []
            for agent in self.agents:
                agent_class_name = agent.agent_node['class_name']
                agent_real_id = agent.agent_node['id']
                if agent.action_history:
                    actions_str = ', '.join(agent.action_history)
                    agent_action_lines.append(f"<{agent_class_name}>({agent_real_id}): {actions_str}")
            
            # Format action history
            action_history = '\n'.join(agent_action_lines)
                
            # Immediately submit next query
            self.submit_async_oracle_query(
                self.oracle_prompt_path,
                obs2text,
                self.env.goal_instruction,
                self.env.num_agent,
                self.dialogue_history,
                action_history,
                self.sampling_params
            )
            print("Oracle next query submitted")

        else:
            if self.async_result is None:
                # No query in flight — submit the initial oracle query
                agent_action_lines = []
                for agent in self.agents:
                    agent_class_name = agent.agent_node['class_name']
                    agent_real_id = agent.agent_node['id']
                    if agent.action_history:
                        actions_str = ', '.join(agent.action_history)
                        agent_action_lines.append(f"<{agent_class_name}>({agent_real_id}): {actions_str}")
                action_history = '\n'.join(agent_action_lines)

                self.submit_async_oracle_query(
                    self.oracle_prompt_path,
                    obs2text,
                    self.env.goal_instruction,
                    self.env.num_agent,
                    self.dialogue_history,
                    action_history,
                    self.sampling_params
                )
                print("Initial oracle query submitted, blocking until done...")

                # Block on the very first oracle call so agents start with a plan
                while not self.oracle_done_flag.value:
                    time.sleep(0.1)
                oracle_message = self.check_async_oracle_result()
                print(f"First oracle result received: {oracle_message}")

                # Immediately submit next async query
                self.submit_async_oracle_query(
                    self.oracle_prompt_path,
                    obs2text,
                    self.env.goal_instruction,
                    self.env.num_agent,
                    self.dialogue_history,
                    action_history,
                    self.sampling_params
                )
            else:
                print("Oracle query in progress, waiting for result...")

        # Process Oracle result if available
        target_class_name = None
        target_real_id = None
        self.subgoal = None
        if oracle_message is not None:
            self.write_log_to_file("Oracle: " + oracle_message)
            self.total_dialogue_history.append((self.env.steps, "Oracle: " + oracle_message))
            message = oracle_message
            self.subgoal = message

            if self.debug:
                print(f"message_oracle_outputs:\n{message}")
            
            try:
                start_class_name = message.find('<') + 1  
                end_class_name = message.find('>')        
                start_id = message.find('(') + 1          
                end_id = message.find(')')                

                # extract class_name and real_id
                target_class_name = message[start_class_name:end_class_name]
                target_real_id = int(message[start_id:end_id])
                
                self.last_instructed_agent_id = target_real_id
                print(f"Oracle selected agent: <{target_class_name}>({target_real_id})")
            except Exception as e:
                print(f"An error occurred while parsing Oracle message: {e}")
                traceback.print_exc()
                error_info = traceback.format_exc()
                self.write_log_to_file(f"An error occurred: {e}")
                self.write_log_to_file(error_info+'\n\n')
                agent_message = "all robot agents: In the last step, the oracle's reasoning was incorrect, and no instructions were given to any of the robot agents, therefore none of the robot agents performed any actions. Please reassess the information in the environment and give a correct instruction strictly following the template 'Hello <class name>(id): #message#.'"
                self.total_dialogue_history.append((self.env.steps, agent_message))
                self.dialogue_history = self._format_dialogue()
        
        # Execute each agent: decide action and execute immediately
        executed_actions_info = []  # Collect all executed actions info
        prev_step = self.env.steps

        for agent_idx, agent in enumerate(self.agents):
            # Get updated observation after previous agents' actions
            obs = self.env.get_observations()
            
            agent_node = agent.agent_node
            agent_real_id = agent_node["id"]
            agent_class_name = agent_node["class_name"]
            agent_obs = self.agent_obs2text(obs, agent_idx)
            # print(f"Agent {agent_class_name}({agent_real_id}) observation: {agent_obs}")
            # Determine prompt path based on agent class
            if agent_class_name == 'quadrotor':
                prompt_path = self.quadrotor_prompt_path
            elif agent_class_name == 'robot dog' or agent_class_name == 'robot_dog':
                prompt_path = self.robot_dog_prompt_path
            elif agent_class_name == 'robot arm' or agent_class_name == 'robot_arm':
                prompt_path = self.robot_arm_prompt_path
            else:
                prompt_path = None
            
            # Only pass instruction if this is the Oracle-selected agent
            instruction = None
            if target_real_id is not None and agent_real_id == target_real_id:
                instruction = oracle_message
                # print(f"Passing Oracle instruction to agent <{agent_class_name}>({agent_real_id})")
            elif target_real_id is not None:
                # Clear last_instruction for other agents
                agent.last_instruction = None
            
            chat_agent_info = {
                "class_name": agent_class_name, 
                "id": agent_real_id, 
                "observation": agent_obs, 
                "instruction": instruction, 
                "prompt_path": prompt_path
            }
            # print(chat_agent_info)
            action, message, info = agent.get_action(obs[agent_idx], chat_agent_info, self.env.task_goal, self.env.goal_instruction)
            print(f"Executing action for <{agent_class_name}>({agent_real_id}): {action}")
            
                # Log all non-wait actions to dialogue history
            if message is not None and self.last_instructed_agent_id is not None and agent_real_id == self.last_instructed_agent_id:
                self.total_dialogue_history.append((prev_step, f"<{agent_class_name}>({agent_real_id}): " + str(message)))
                self.dialogue_history = self._format_dialogue()
            
            # If agent decided an action, execute it immediately
            if action is not None:

                # Execute action immediately
                try:
                    done, task_results, satisfied, unsatisfied, _ = self.env.step(agent_class_name, agent_real_id, action, self.task_goal)
                    
                    self.last_done = done
                    self.last_task_results = task_results
                    self.last_satisfied = satisfied
                    self.last_unsatisfied = unsatisfied
                    
                    # Collect executed action info
                    executed_actions_info.append({
                        'agent_idx': agent_idx,
                        'agent_class_name': agent_class_name,
                        'agent_real_id': agent_real_id,
                        'action': action,
                        'message': message,
                        'had_instruction': instruction if instruction is not None else None
                    })
                    
                except Exception as e:
                    print(f"Exception occurs when performing action for agent {agent_real_id}: {action}")
                    traceback.print_exc()
                    raise Exception
        
        # If no actions were executed
        if len(executed_actions_info) == 0:
            done = self.last_done
            task_results = self.last_task_results
            satisfied = self.last_satisfied
            unsatisfied = self.last_unsatisfied
        
        # Set step counter: increment by 1 regardless of how many agents acted
        if len(executed_actions_info) != 0:
            steps = prev_step + 1
        else:
            # Wait for any agent's LLM or oracle's LLM to complete before continuing
            steps = prev_step
            while not (any(agent.llm_done_flag.value for agent in self.agents) or self.oracle_done_flag.value):
                time.sleep(1)  # Short sleep to avoid busy waiting
        
        # Synchronize step counter across environment and all agents
        self.env.steps = steps
        for agent in self.agents:
            agent.steps = steps
        print(self.dialogue_history)
        return done, task_results, satisfied, unsatisfied, executed_actions_info, steps


    def run(self):
        
        self.task_goal = copy.deepcopy(self.env.task_goal)
        saved_info = []

        success = False
        count = 0
        self.start_time = time.time()
        while True:
            count += 1
            done, task_results, satisfied, unsatisfied, executed_actions_info, steps = self.step()
            
            # Extract lists from executed_actions_info
            agent_ids = [info['agent_idx'] for info in executed_actions_info]
            actions = [info['action'] for info in executed_actions_info]
            agent_messages = [info['message'] for info in executed_actions_info]
            
            saved_info.append({'task_id': self.env.task_id,
                        'env_id': self.env.env_id,
                        'task_name': self.env.task_name,
                        'gt_steps': self.env.ground_truth_step_num,
                        'task_goal': self.task_goal,
                        'goal_instruction': self.env.goal_instruction,
                        'step': steps,
                        'subgoal': self.subgoal,
                        'agent_ids': agent_ids,  # List of agent indices
                        'actions': actions,  # List of actions
                        'agent_messages': agent_messages,  # List of messages
                        'satisfied': satisfied,
                        'unsatisfied': unsatisfied,
                        'env_graph': self.env.graph, 
                })
           
            success = done
      
            max_setp = 2 * self.env.ground_truth_step_num
            if (self.oracle_count > max_setp):
                print("---------------------------")
                print("The task failed, exceeding 2 times the number of GT steps")
                print(f"Whether steps in gt*2+1 are successful:{done}")
                print(f" setps: {steps}/{max_setp}")
                print("---------------------------")
                # Record final result
                elapsed = time.time() - getattr(self, 'start_time', time.time())
                
                # Collect action history from all agents
                agent_histories = []
                for idx, agent in enumerate(self.agents):
                    agent_class = agent.agent_node['class_name']
                    agent_id = agent.agent_node['id']
                    history = agent.action_history if hasattr(agent, 'action_history') else []
                    agent_histories.append(f"  <{agent_class}>({agent_id}): {history}")
                
                final_log = (
                    f"FINAL RESULT - Task_ID: {self.env.task_id}, Task_Name: {self.env.task_name}, "
                    f"Num_Agents: {self.num_agents}, Task_Goal: {self.task_goal}, Goal_Instruction: {self.env.goal_instruction},\n "
                    f"Success: False, Satisfied: {satisfied}, Unsatisfied: {unsatisfied}, Elapsed_Time: {elapsed:.2f}s\n"
                    f"Agent Action Histories:\n" + "\n".join(agent_histories)
                )
                self.write_action_to_file(final_log)
                self.write_log_to_file(f'''---------------------------
The task failed, exceeding 2 times the number of GT steps
Whether steps in gt*2+1 are successful:{done}
setps: {steps}/{max_setp}
oracle_count: {self.oracle_count}/{max_setp}
---------------------------
''')
            
                success = False
                break
            
            if success:
                
                self.write_log_to_file(f'''-------------------------------------
success!
setps: {steps}/{max_setp}
oracle_count: {self.oracle_count}/{max_setp}
--------------------------------
''')
                # Record final result with elapsed time on success as well
                elapsed = time.time() - getattr(self, 'start_time', time.time())
                
                # Collect action history from all agents
                agent_histories = []
                for idx, agent in enumerate(self.agents):
                    agent_class = agent.agent_node['class_name']
                    agent_id = agent.agent_node['id']
                    history = agent.action_history if hasattr(agent, 'action_history') else []
                    agent_histories.append(f"  <{agent_class}>({agent_id}): {history}")
                
                final_log = (
                    f"FINAL RESULT - Task_ID: {self.env.task_id}, Task_Name: {self.env.task_name}, "
                    f"Num_Agents: {self.num_agents}, Task_Goal: {self.task_goal}, Goal_Instruction: {self.env.goal_instruction},\n "
                    f"Success: True, Satisfied: {satisfied}, Unsatisfied: {unsatisfied}, Elapsed_Time: {elapsed:.2f}s\n"
                    f"Agent Action Histories:\n" + "\n".join(agent_histories)
                )
                self.write_action_to_file(final_log)
                break
        saved_info[steps-1]['is_finished'] = success        
        return success, steps, saved_info

    def update_dict(self,key, value, my_dict):

        my_dict[key] = value
        return my_dict