
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
		
		# Store results in shared dict (oracle_message is now the final formatted message)
		result_dict['oracle_message'] = oracle_message
		done_flag.value = True
		write_oracle_log_to_file(f"Oracle Response: {oracle_message}")
		return oracle_message
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
        self.total_dialogue_history = []
        self.chat = True
        self.source = args.source
        self.lm_id = args.lm_id
        # Logging: results/<branch>/<lm_id>/<env>/
        self.result_dir = f'./results/{args.branch}/{self.lm_id}/{args.env}'
        os.makedirs(self.result_dir, exist_ok=True)
        self.record_dir = os.path.join(self.result_dir, f'{args.env}.txt')
        self.device = None
        self.sampling_parameters = None
        self.total_cost = 0

        self.last_done = False
        self.last_task_results = None
        self.last_satisfied = None
        self.last_unsatisfied = None
        self.costdict = {}
        self.mesage_target= {}
        self.subgoal = None
        self.max_oracle_step = -1  # Track the maximum oracle step received
        self.count = 0
        self.oracle_count = 0
        if self.source == 'openai':
            api_key = args.api_key # your openai api key
            organization = args.organization # your openai organization

            # Create OpenAI client for sync oracle calls
            self.client = OpenAI(api_key=api_key, organization=organization)

            if self.chat:
                # GPT-5 only - no temperature or max_tokens support
                self.sampling_params = {
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
            self.real_id = None
            self.class_name = None
            self.id = None
            self.transition = False
            # Oracle async management - 1 slot
            self.async_results = [None]
            self.oracle_done_flags = [self.manager.Value('b', False)]
            self.oracle_result_dicts = [self.manager.dict()]
            self.oracle_issue_steps = [None]
        else:
            raise ValueError("Only 'openai' source is currently supported")

    def get_actions(self, obs, chat_agent_info, steps):

        for id, agent in enumerate(self.agents):
            if agent.agent_node["id"] == chat_agent_info["id"]:

                action, message, info = agent.get_action(obs[id], chat_agent_info, self.env.task_goal, steps)

        return action, message, info
    
    def submit_async_oracle_query(self, oracle_prompt_path, obs2text, goal_instruction, num_agent, dialogue_history, action_history, sampling_params, target_step=None):
        """Submit async oracle query to the first available slot
        
        Args:
            oracle_prompt_path: Path to oracle prompt template file
            obs2text: Observation text
            goal_instruction: Task goal instruction
            num_agent: Number of agents
            dialogue_history: Dialogue history string
            action_history: Action history string
            sampling_params: Sampling parameters
            target_step: The step number to compute oracle message for. If None, uses self.env.steps
        """
        if not self.use_pool:
            raise ValueError("Pool not initialized")
        
        # Use single slot (index 0)
        if self.async_results[0] is not None:
            print("Oracle: Slot busy, waiting...")
            return
        
        if target_step is None:
            target_step = self.env.steps
        
        # Record issue step
        self.oracle_issue_steps[0] = target_step
        
        # Reset shared memory for new query
        self.oracle_done_flags[0].value = False
        self.oracle_result_dicts[0].clear()
        
        # Submit to process pool
        self.async_results[0] = self.oracle_pool.apply_async(
            _oracle_worker_generate_single,
            (oracle_prompt_path, obs2text, goal_instruction, num_agent, dialogue_history, action_history, sampling_params, self.oracle_done_flags[0], self.oracle_result_dicts[0])
        )
    
    def check_async_oracle_result(self):
        """Check async oracle query and write completed result to self.mesage_target"""
        # Check if the single slot is completed
        if self.async_results[0] is not None and self.oracle_done_flags[0].value:
            issue_step = self.oracle_issue_steps[0]
            
            # Get the result
            self.async_results[0].get(timeout=0.1)
            oracle_message = self.oracle_result_dicts[0].get('oracle_message', None)
            
            # Write to self.mesage_target
            self.mesage_target[issue_step] = oracle_message
            self.oracle_count += 1
            
            # Update max oracle step
            if issue_step > self.max_oracle_step:
                self.max_oracle_step = issue_step
            
            # Clear this slot
            self.async_results[0] = None
            self.oracle_done_flags[0].value = False
            self.oracle_issue_steps[0] = None
    
    def wait_for_oracle_result(self):
        """Wait for oracle result by polling until slot is available"""
        if self.async_results[0] is not None:
            while True:
                self.check_async_oracle_result()
                if self.async_results[0] is None:
                    break
                time.sleep(1)
        else:
            self.check_async_oracle_result()
    
    def __del__(self):
        """Cleanup the process pool when the object is destroyed"""
        if hasattr(self, 'oracle_pool'):
            self.oracle_pool.close()
            self.oracle_pool.join()
        if hasattr(self, 'manager'):
            self.manager.shutdown()

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
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')

    def step(self):
        # Wait if: (1) first oracle result needed, OR (2) slot is busy
        obs = self.env.get_observations()
        id_name_dict = self.env.id_name_dict

        obs2text = ''
        for i in range(self.num_agents):
            obs2text += self.agent_obs2text(obs, i) + '\n'

        oracle_prompt_path = self.oracle_prompt_path
        
        agent_action_lines = []
        for agent in self.agents:
            agent_class_name = agent.agent_node['class_name']
            agent_real_id = agent.agent_node['id']
            if agent.action_history:
                sorted_history = sorted(agent.action_history.items())[-10:]
                # Convert None actions to [wait]
                actions_str = ', '.join([f"{v if v is not None else '[wait]'} at step {k}" for k, v in sorted_history])
                agent_action_lines.append(f"<{agent_class_name}>({agent_real_id}): {actions_str}")
        action_history = '\n'.join(agent_action_lines) if agent_action_lines else 'None'
        # Submit query for current step if not already available
        
        self.wait_for_oracle_result()
        if self.env.steps not in self.mesage_target:
            self.submit_async_oracle_query(
                oracle_prompt_path,
                obs2text,
                self.env.goal_instruction,
                self.env.num_agent,
                self.dialogue_history,
                action_history,
                self.sampling_params,
                target_step=self.env.steps
            )
            self.wait_for_oracle_result()
        
        # Submit next step query
        self.submit_async_oracle_query(
            oracle_prompt_path,
            obs2text,
            self.env.goal_instruction,
            self.env.num_agent,
            self.dialogue_history,
            action_history,
            self.sampling_params,
            target_step=self.env.steps + 1
        )

        # Get oracle message from mesage_target
        oracle_message = self.mesage_target.get(self.env.steps)
        
        # Store oracle message (no separate draft)
        self.subgoal = oracle_message
        start_class_name = self.subgoal.find('<') + 1  
        end_class_name = self.subgoal.find('>')        
        start_id = self.subgoal.find('(') + 1          
        end_id = self.subgoal.find(')')
        # extract class_name and real_id
        real_id = int(oracle_message[start_id:end_id])
        if real_id != self.real_id:
            self.transition = True
            # Wait for all pending async results of the previous agent
            if self.real_id is not None:
                prev_agent = None
                for agent in self.agents:
                    if agent.agent_node['id'] == self.real_id:
                        prev_agent = agent
                        break
                
                if prev_agent is not None:
                    # Wait until slot is completed
                    while prev_agent.async_results[0] is not None:
                        prev_agent.check_async_llm_result()
                        time.sleep(1)         

        self.class_name = self.subgoal[start_class_name:end_class_name]
        self.real_id = real_id
        self.id  = [key for key, value in id_name_dict.items() if value[1] == self.real_id]

        self.total_dialogue_history.append("Oracle: " + oracle_message)
        print(f"Oracle result received (step {self.env.steps}): {oracle_message}")

        try:
            agent_obs = self.agent_obs2text(obs, self.id[0])

            # Initialize prompt_path to avoid UnboundLocalError
            prompt_path = None
            
            if self.class_name == 'quadrotor':
                prompt_path = self.quadrotor_prompt_path
            elif self.class_name == 'robot dog' or self.class_name == 'robot_dog':
                prompt_path = self.robot_dog_prompt_path
            elif self.class_name == 'robot arm' or self.class_name == 'robot_arm':
                prompt_path = self.robot_arm_prompt_path
            else:
                # Log unexpected class name and skip this step
                print(f"Warning: Unexpected class_name '{self.class_name}' extracted from oracle message")
                raise ValueError(f"Unknown class_name: {self.class_name}")

            chat_agent_info = {"class_name": self.class_name, 
                            "id": self.real_id, 
                            "observation": agent_obs, 
                            "instruction": self.subgoal, 
                            "prompt_path": prompt_path
                            }

            agent_action, agent_message, agent_info = self.get_actions(obs,  chat_agent_info,self.env.steps)

            self.total_dialogue_history.append(f"<{self.class_name}>({self.real_id}): " + str(agent_message))
            numbered_list = [f"[{i+1}]、{item}" for i, item in enumerate(self.total_dialogue_history)]
            self.dialogue_history = '\n'.join(numbered_list[-10:])
        except Exception as e:
    
            print(f"An error occurred: {e}")
            traceback.print_exc()
            error_info = traceback.format_exc()
            
            # Log the error
            self.write_log_to_file(f"\n[ERROR] Agent execution failed at step {self.env.steps}")
            self.write_log_to_file(f"Error type: {type(e).__name__}")
            self.write_log_to_file(f"Error message: {str(e)}")
            self.write_log_to_file(f"Traceback:\n{error_info}")

            agent_action = None
            agent_message = "all robot agents: In the last step, the oracle's reasoning was incorrect, and no instructions were given to any of the robot agents, therefore none of the robot agents performed any actions. Please reassess the information in the environment and give a correct instruction strictly following the template 'Hello <class name>(id): #message#.'"
            self.total_dialogue_history.append(agent_message)
            numbered_list = [f"[{i+1}]、{item}" for i, item in enumerate(self.total_dialogue_history)]
            self.dialogue_history = '\n'.join(numbered_list[-10:])
        
        if agent_action is None:
            done = self.last_done
            task_results = self.last_task_results
            satisfied = self.last_satisfied
            unsatisfied = self.last_unsatisfied
            self.env.steps += 1
        else:
            try:
                done, task_results,satisfied, unsatisfied,steps= self.env.step(self.class_name, self.real_id, agent_action, self.task_goal)
                self.last_done = done
                self.last_task_results = task_results
                self.last_satisfied = satisfied
                self.last_unsatisfied = unsatisfied
                self.count+=1
                
            except Exception as e:
                print("Exception occurs when performing action: ", agent_action)
                raise Exception
        self.write_log_to_file(f'\nDIALOGUE_HISTORY:\n{self.dialogue_history}')  
        steps = self.env.steps
        return done, task_results, satisfied, unsatisfied, id, agent_action, agent_message,steps,self.count


    def run(self):
        
        self.task_goal = copy.deepcopy(self.env.task_goal)
        saved_info = []

        success = False
        count = 0
        self.start_time = time.time()
        while True:
            done, task_results, satisfied, unsatisfied, id, agent_action, agent_message,steps,count  = self.step()
            
            saved_info.append({'task_id': self.env.task_id,
                        'env_id': self.env.env_id,
                        'task_name': self.env.task_name,
                        'gt_steps': self.env.ground_truth_step_num,
                        'task_goal': self.task_goal,
                        'goal_instruction': self.env.goal_instruction,
                        'step': steps,
                        'subgoal': self.subgoal,
                        'agent_ids': id,  # List of agent indices
                        'actions': agent_action,  # List of actions
                        'agent_messages': agent_message,  # List of messages
                        'satisfied': satisfied,
                        'unsatisfied': unsatisfied,
                        'env_graph': self.env.graph, 
                })
            
            success = done
      
            max_setp = 2 * self.env.ground_truth_step_num
            if self.oracle_count > max_setp:
                print("---------------------------")
                print("The task failed, exceeding 2 times the number of GT steps")
                print(f"Whether steps in gt*2+1 are successful:{done}")
                print(f" setps: {steps}/{max_setp}")
                print(f"oracle_count: {self.oracle_count}/{max_setp}")
                print("---------------------------")
                # Record final result
                elapsed = time.time() - getattr(self, 'start_time', time.time())
                
                # Collect action history from all agents
                agent_histories = []
                for idx, agent in enumerate(self.agents):
                    agent_class = agent.agent_node['class_name']
                    agent_id = agent.agent_node['id']
                    history = agent.action_history if hasattr(agent, 'action_history') else {}
                    agent_histories.append(f"  <{agent_class}>({agent_id}): {history}")
                
                final_log = (
                    f"FINAL RESULT - Task_ID: {self.env.task_id}, Task_Name: {self.env.task_name}, "
                    f"Num_Agents: {self.num_agents}, Task_Goal: {self.task_goal}, Goal_Instruction: {self.env.goal_instruction},\n "
                    f"Success: False, Satisfied: {satisfied}, Unsatisfied: {unsatisfied}, Elapsed_Time: {elapsed:.2f}s\n"
                    f"Agent Action Histories:\n" + "\n".join(agent_histories)
                )
                self.write_log_to_file(final_log)
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
---------------------------------
''')
                # Record final result with elapsed time on success as well
                elapsed = time.time() - getattr(self, 'start_time', time.time())
                
                # Collect action history from all agents
                agent_histories = []
                for idx, agent in enumerate(self.agents):
                    agent_class = agent.agent_node['class_name']
                    agent_id = agent.agent_node['id']
                    history = agent.action_history if hasattr(agent, 'action_history') else {}
                    agent_histories.append(f"  <{agent_class}>({agent_id}): {history}")
                
                final_log = (
                    f"FINAL RESULT - Task_ID: {self.env.task_id}, Task_Name: {self.env.task_name}, "
                    f"Num_Agents: {self.num_agents}, Task_Goal: {self.task_goal}, Goal_Instruction: {self.env.goal_instruction},\n "
                    f"Success: True, Satisfied: {satisfied}, Unsatisfied: {unsatisfied}, Elapsed_Time: {elapsed:.2f}s\n"
                    f"Agent Action Histories:\n" + "\n".join(agent_histories)
                )
                self.write_log_to_file(final_log)
                break
        saved_info[steps-1]['is_finished'] = success        
        return success, steps, saved_info

    def update_dict(self,key, value, my_dict):

        my_dict[key] = value
        return my_dict