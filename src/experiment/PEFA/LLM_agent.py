from LLM import *
import re
import copy
import time
from multiprocessing import Pool, Manager

_worker_agent_llm = None

def _pool_init_agent_llm(init_args):
	"""Initialize single LLM instance in worker process"""
	global _worker_agent_llm
	source, lm_id, args, agent_node = init_args
	_worker_agent_llm = LLM(source, lm_id, args, agent_node)


def _pool_check_init():
	"""Dummy function to check if pool initialization is complete"""
	global _worker_agent_llm
	return _worker_agent_llm is not None

def async_agent_llm_worker(llm_args, done_flag, result_plan, result_info):
	"""Worker function that runs agent LLM planning in a separate process"""
	try:
		global _worker_agent_llm
		
		# Unpack arguments (same order as LLM_plan)
		(agent_node, chat_agent_info, current_room, next_rooms,
		 all_landable_surfaces, landable_surfaces, on_surfaces,
		 grabbed_objects, reachable_objects, unreached_objects,
		 on_same_surfaces, action_history) = llm_args
		
		# Use the shared global LLM instance
		message, info, raw_output = _worker_agent_llm.run(
			agent_node, chat_agent_info, current_room, next_rooms,
			all_landable_surfaces, landable_surfaces, on_surfaces,
			grabbed_objects, reachable_objects, unreached_objects,
			on_same_surfaces, action_history
		)
		
		# Store results in shared memory
		result_plan['plan'] = info.get('plan', None)
		result_plan['message'] = message
		result_plan['raw_output'] = raw_output
		result_info.update(info)
		
		# Set done flag
		done_flag.value = True
		
	except Exception as e:
		print(f"Error in async agent LLM worker: {e}")
		import traceback
		traceback.print_exc()
		result_plan['plan'] = None
		result_plan['message'] = None
		result_info['error'] = str(e)
		done_flag.value = True

class LLM_agent:
	"""
	LLM agent class
	"""
	def __init__(self, agent_id, args, agent_node, init_graph):

		self.agent_node = agent_node
		self.agent_id = agent_id
		self.init_graph = init_graph
		self.init_id2node = {x['id']: x for x in init_graph['nodes']}
		self.source = args.source
		self.lm_id = args.lm_id
		self.args = args
		self.LLM = LLM(self.source, "gpt-5-nano", self.args, agent_node)
		self.unsatisfied = {}
		self.steps = 0
		self.plan = None
		self.current_room = None
		self.grabbed_objects = None
		self.goal_location = None
		self.goal_location_id = None
		self.last_action = None
		self.id2node = {}
		self.id_inside_room = {}
		self.satisfied = []
		self.reachable_objects = []
		self.action_history = {}

		agent_init_args = (self.source, self.lm_id, self.args, agent_node)
		self.agent_pool = Pool(
			processes=1,
			initializer=_pool_init_agent_llm,
			initargs=(agent_init_args,)
		)
		self.plan_target = {}
		# Wait for pool initialization to complete
		self.agent_pool.apply(_pool_check_init)
		
		self.manager = Manager()
		
		# Async LLM call management - 1 slot
		self.async_results = [None]
		self.llm_done_flags = [self.manager.Value('b', False)]
		self.async_plans = [self.manager.dict()]
		self.async_infos = [self.manager.dict()]
		self.issue_steps = [None]

	def LLM_plan(self):
		message, info, raw_output = self.LLM.run(self.agent_node, self.chat_agent_info, self.current_room, self.next_rooms, self.all_landable_surfaces,self.landable_surfaces, 
					  self.on_surfaces, self.grabbed_objects, self.reachable_objects, self.unreached_objects, self.on_same_surfaces, self.action_history)
		return message, info, raw_output


	def check_progress(self, state, goal_spec):
		unsatisfied = {}
		satisfied = []
		id2node = {node['id']: node for node in state['nodes']}

		for key, value in goal_spec.items():
			elements = key.split('_')
			self.goal_location_id = int((re.findall(r'\((.*?)\)', elements[-1]))[0])
			self.target_object_id = int((re.findall(r'\((.*?)\)', elements[1]))[0])
			cnt = value[0]
			for edge in state['edges']:
				if cnt == 0:
					break
				if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and edge['from_id'] == self.target_object_id:
					satisfied.append(id2node[edge['from_id']])  # A list of nodes that meet the goal
					cnt -= 1
					# if self.debug:
					# 	print(satisfied)
			if cnt > 0:
				unsatisfied[key] = value  
		return satisfied, unsatisfied

	def submit_async_llm_query(self, target_step=None, use_predict_prompt=False):
		"""Submit async LLM query to the single slot
		
		Args:
			target_step: The step number to compute plan for. If None, uses self.steps
			use_predict_prompt: If True, use predict prompt instead of normal prompt
		"""
		# Use single slot (index 0)
		if self.async_results[0] is not None:
			# print(f"Agent {self.agent_node['class_name']}({self.agent_node['id']}): Slot busy")
			return
		
		if target_step is None:
			target_step = self.steps
		
		# Record issue step
		self.issue_steps[0] = target_step
		
		# Reset shared memory for new query
		self.llm_done_flags[0].value = False
		self.async_plans[0].clear()
		self.async_infos[0].clear()
		
		# Use chat_agent_info directly
		chat_agent_info_copy = dict(self.chat_agent_info)
		
		# Add use_predict_prompt to chat_agent_info
		if use_predict_prompt:
			chat_agent_info_copy['use_predict_prompt'] = True
		
		# Prepare arguments tuple
		llm_args = (
			self.agent_node, chat_agent_info_copy, self.current_room, self.next_rooms,
			self.all_landable_surfaces, self.landable_surfaces, self.on_surfaces,
			self.grabbed_objects, self.reachable_objects, self.unreached_objects,
			self.on_same_surfaces, self.action_history
		)
		
		# Submit to process pool
		self.async_results[0] = self.agent_pool.apply_async(
			async_agent_llm_worker,
			(llm_args, self.llm_done_flags[0], self.async_plans[0], self.async_infos[0])
		)
	
	def check_async_llm_result(self):
		"""Check async LLM query and write completed result to self.plan_target"""
		# Check if the single slot is completed
		if self.async_results[0] is not None and self.llm_done_flags[0].value:
			issue_step = self.issue_steps[0]
			
			# Get the result
			self.async_results[0].get(timeout=0.1)
			plan = self.async_plans[0].get('plan', None)
			message = self.async_plans[0].get('message', None)
			raw_output = self.async_plans[0].get('raw_output', None)
			
			# Write to self.plan_target (plan, message, and raw_output)
			self.plan_target[issue_step] = {'plan': plan, 'message': message, 'raw_output': raw_output}
			
			# Clear this slot
			self.async_results[0] = None
			self.llm_done_flags[0].value = False
			self.issue_steps[0] = None
	
	def wait_for_llm_result(self):
		"""Wait for LLM result by polling until slot is available"""
		if self.async_results[0] is not None:
			while True:
				self.check_async_llm_result()
				if self.async_results[0] is None:
					break
				time.sleep(1)
		else:
			self.check_async_llm_result()

	def get_action(self, observation, chat_agent_info, goal, steps):
		self.steps = steps
		self.chat_agent_info = chat_agent_info
		satisfied, unsatisfied = self.check_progress(observation, goal) 
		# print(f"satisfied: {satisfied}")
		if len(satisfied) > 0:
			self.unsatisfied = unsatisfied
			self.satisfied = satisfied

		obs = observation
		self.grabbed_objects = None
		self.reachable_objects = []
		self.landable_surfaces = None
		self.on_surfaces = None
		self.all_landable_surfaces = []
		self.all_landable_surfaces = [x for x in obs['nodes'] if 'LANDABLE' in x['properties']]
		self.on_same_surfaces = []
		self.on_same_surfaces_ids = []
		self.chat_agent_info = chat_agent_info

		self.id2node = {x['id']: x for x in obs['nodes']}

		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			
			if x == self.agent_node['id']:

				if r == 'INSIDE':
					self.current_room = self.id2node[y]
				if r == 'ON' :
					self.on_surfaces = self.id2node[y]
					if self.agent_node['class_name'] == 'robot arm' or self.agent_node['class_name'] == 'robot_arm':
						for i in range(3):
							for edge in obs['edges']:
								if (edge['from_id'] != x and edge['to_id'] == y and edge['relation_type'] == 'ON') or (edge['from_id'] != x and edge['to_id'] in self.on_same_surfaces_ids and edge['relation_type'] == 'ON') or (edge['from_id'] != x and edge['to_id'] in self.on_same_surfaces_ids and edge['relation_type'] == 'INSIDE') :
									self.on_same_surfaces_ids.append(edge['from_id'])
									#self.on_same_surfaces.append(self.id2node[edge['from_id']])  # Find any contain or surface on the table
									if 'SURFACES' in self.id2node[edge['from_id']]['properties'] or 'CONTAINERS' in self.id2node[edge['from_id']]['properties']:
										for ee in obs['edges']:
											if ee['to_id'] == edge['from_id'] and (ee['relation_type'] == 'INSIDE' or ee['relation_type'] == 'ON'):
												self.on_same_surfaces_ids.append(ee['from_id'])
												#self.on_same_surfaces.append(self.id2node[ee['from_id']]) # The goal here is to find objects that are not directly on the surface
								self.on_same_surfaces_ids = list(set(self.on_same_surfaces_ids))	
						for id in self.on_same_surfaces_ids:
							self.on_same_surfaces.append(self.id2node[id])
				
				if r == 'HOLD':
					# self.grabbed_objects.append(y)
					self.grabbed_objects = self.id2node[y]
				if r == 'CLOSE':
					self.reachable_objects.append(self.id2node[y])
				if r == 'ABOVE' and 'LANDABLE' in self.id2node[y]['properties']:
					self.landable_surfaces = self.id2node[y]

		self.unreached_objects = copy.deepcopy(obs['nodes'])
		for node in obs['nodes']:
			if node == self.grabbed_objects or node in self.reachable_objects:
				self.unreached_objects.remove(node)
			elif node['category'] == 'Rooms' or node['category'] == 'Agents' or node['category'] == 'Floor' or "HIGH_HEIGHT" in node['properties'] or 'ON_HIGH_SURFACE' in node['properties']:
				self.unreached_objects.remove(node)  #The idea here is to find the places that the robotic dog has not reached, remove what it already has in its hand, remove what it is close to, remove the room, the floor, the agent itself, the high surface and what is on the high surface

		self.doors = []
		self.next_rooms = []
		self.doors = [x for x in obs['nodes'] if x['class_name'] == 'door']
		for door in self.doors:
			for edge in self.init_graph['edges']:
				if edge['relation_type'] == "LEADING TO" and edge['from_id'] == door['id'] and edge['to_id'] != self.current_room["id"]:
						self.next_rooms.append([self.init_id2node[edge['to_id']], door])

		info = {'graph': obs,
				"obs": {	
						 "agent_class": self.agent_node["class_name"],
						 "agent_id":self.agent_node["id"],
						 "grabbed_objects": self.grabbed_objects,
						 "reachable_objects": self.reachable_objects,
						 "on_surfaces": self.on_surfaces,
						 "landable_surfaces": self.landable_surfaces,
						 "doors": self.doors,
						 "next_rooms": self.next_rooms,
						 "objects_on_the_same_surfaces": self.on_same_surfaces,
						 "satisfied": self.satisfied,
						 "current_room": self.current_room['class_name'],
						},
				}
		self.wait_for_llm_result()
		if self.steps not in self.plan_target:
			self.submit_async_llm_query(target_step=self.steps, use_predict_prompt=False)
			self.wait_for_llm_result()
		
		_, _, available_plans = self.LLM.get_available_plans(
			self.agent_node, self.next_rooms, self.all_landable_surfaces,
			self.landable_surfaces, self.on_surfaces, self.grabbed_objects,
			self.reachable_objects, self.unreached_objects, self.on_same_surfaces
		)
		if self.plan_target[self.steps]['plan'] not in available_plans:
			raw_output = self.plan_target[self.steps]['raw_output']
			# Try to parse the raw output with available_plans
			parsed_plan, sorry_msg = self.LLM.parse_single_output(raw_output, available_plans)
			if parsed_plan is not None:
				# Update plan_target with parsed result
				self.plan_target[self.steps]['plan'] = parsed_plan
				# Create proper message for successful action
				self.plan_target[self.steps]['message'] = f"ACTION: {parsed_plan}"
			elif sorry_msg and '[wait]' in sorry_msg.lower():
				# Handle [wait] action (plan=None but valid wait action)
				self.plan_target[self.steps]['plan'] = None
				self.plan_target[self.steps]['message'] = "ACTION: [wait]"
			else:
				# Re-query if parsing failed
				self.submit_async_llm_query(target_step=self.steps, use_predict_prompt=False)
				self.wait_for_llm_result()
		
		# Use predict prompt for future step if use_futures is enabled
		self.submit_async_llm_query(target_step=self.steps+1, use_predict_prompt=True)

		if self.plan_target[self.steps] is None: 
			print("No more things to do!")
		plan = self.plan_target[self.steps]['plan']
		self.action_history[self.steps] = plan
		message = self.plan_target[self.steps]['message']
		
		return plan, message, info
