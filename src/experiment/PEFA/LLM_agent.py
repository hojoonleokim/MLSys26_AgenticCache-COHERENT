from LLM import *
from action_cache import ActionCache
import re
import copy
import torch
from multiprocessing import Pool, Manager
import time
# Global variable for worker process - single LLM instance shared across all agents
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
		message, info = _worker_agent_llm.run(
			agent_node, chat_agent_info, current_room, next_rooms,
			all_landable_surfaces, landable_surfaces, on_surfaces,
			grabbed_objects, reachable_objects, unreached_objects,
			on_same_surfaces, action_history
		)
		
		# Store results in shared memory
		result_plan['plan'] = info.get('plan', None)
		result_plan['message'] = message
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
		self._LLM = LLM(self.source, self.lm_id, self.args, self.agent_node)
		self.unsatisfied = {}
		self.steps = 0
		self.current_room = None
		self.grabbed_objects = None
		self.goal_location = None
		self.goal_location_id = None
		self.last_action = None
		self.last_instruction = None
		self.instruction_queue = []
		self.id2node = {}
		self.id_inside_room = {}
		self.satisfied = []
		self.reachable_objects = []
		self.arrived = False
		self.action_history = []  # Store history of executed actions
		self.step_metadata = {}   # {step_num: {metadata dict}} for rich JSONL logging
		self._pending_new_instruction = False
		self._current_available_plans = []
		# Initialize action cache with agent type
		agent_type = agent_node['class_name'].lower().replace(' ', '_')
		self.action_cache = ActionCache(agent_type)
		
		# Create agent pool with initialization
		agent_init_args = (self.source, self.lm_id, self.args, agent_node)
		self.agent_pool = Pool(
			processes=1,
			initializer=_pool_init_agent_llm,
			initargs=(agent_init_args,)
		)
		
		# Wait for pool initialization to complete
		self.agent_pool.apply(_pool_check_init)
		
		self.manager = Manager()
		
		# Async LLM call management
		self._llm_query_prev_action = None  # prev_action saved at LLM query time
		self._llm_query_new_segment = False  # new_instruction_received saved at LLM query time
		self._llm_query_available_actions = []  # available_plans saved at LLM query time
		self._llm_query_last_instruction = None  # last_instruction saved at LLM query time
		self._llm_query_grabbed_objects = None  # grabbed_objects saved at LLM query time
		self._llm_query_basket_contents = None  # basket_contents saved at LLM query time
		self.async_result = None
		self.llm_done_flag = self.manager.Value('b', False)
		self.async_plan = self.manager.dict()
		self.async_info = self.manager.dict()

	def submit_async_llm_query(self):
		"""Submit async LLM query using agent pool"""
		if self.async_result is not None:
			# print(f"Agent {self.agent_node['class_name']}({self.agent_node['id']}): Async LLM already started")
			return
		# Save context at query time for cache update
		self._llm_query_prev_action = self.last_action
		self._llm_query_new_segment = self._pending_new_instruction
		self._llm_query_available_actions = list(self._current_available_plans)
		self._llm_query_last_instruction = self.last_instruction
		self._llm_query_grabbed_objects = self.grabbed_objects
		self._llm_query_basket_contents = self._cache_basket_contents
		self._llm_query_agents_on_same_surface = list(self._cache_agents_on_same_surface)
		self._llm_query_agents_in_room = list(self._cache_agents_in_room)
		self._llm_query_agents_on_land_surface = list(self._cache_agents_on_land_surface)
		self._llm_query_unsatisfied = dict(self.unsatisfied) if self.unsatisfied else {}
		self._llm_query_objects_on_same_surface = list(self._cache_objects_on_same_surface)
		self._llm_query_agent_surface_ids = set(self._cache_agent_surface_ids)
		self._llm_query_obs = self.chat_agent_info.get('observation', '') if hasattr(self, 'chat_agent_info') else ''
		self._pending_new_instruction = False
		# Reset shared memory for new query
		self.llm_done_flag.value = False
		self.async_plan.clear()
		self.async_info.clear()
		
		# Use last_instruction instead of chat_agent_info['instruction'] to avoid None
		chat_agent_info_copy = dict(self.chat_agent_info)
		chat_agent_info_copy['instruction'] = self.last_instruction
		
		# Prepare arguments tuple
		llm_args = (
			self.agent_node, chat_agent_info_copy, self.current_room, self.next_rooms,
			self.all_landable_surfaces, self.landable_surfaces, self.on_surfaces,
			self.grabbed_objects, self.reachable_objects, self.unreached_objects,
			self.on_same_surfaces, self.action_history
		)
		# Submit to process pool
		self.async_result = self.agent_pool.apply_async(
			async_agent_llm_worker,
			(llm_args, self.llm_done_flag, self.async_plan, self.async_info)
		)
	
	def check_async_llm_result(self):
		"""Check if async LLM query is complete and return results"""
		if self.llm_done_flag.value and self.async_result is not None:
			# Get the result (this should be immediate since done_flag is True)
			self.async_result.get(timeout=0.1)
			plan = self.async_plan.get('plan', None)
			message = self.async_plan.get('message', None)
			info = dict(self.async_info)
			
			# Clear the async result
			self.async_result = None
			self.llm_done_flag.value = False
			return plan, message, info
		
		return None, None, {}

	def LLM_plan(self):

		return self.LLM.run(self.agent_node, self.chat_agent_info, self.current_room, self.next_rooms, self.all_landable_surfaces,self.landable_surfaces, 
					  self.on_surfaces, self.grabbed_objects, self.reachable_objects, self.unreached_objects, self.on_same_surfaces)


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

	def instruction_parser(self, instruction):
		"""
		Parse Oracle instruction into a list of standardized actions.
		Steps:
		1. Remove everything after 'so' (reasoning part)
		2. Split by separators: ';', 'then', ','
		3. Extract action and objects in standard format
		Returns:
			List of action strings in format: [action] <object>(id) ...
		"""
		import re

		# Step 1: Remove greeting and 'so' reasoning part
		if ':' in instruction:
			instruction = instruction.split(':', 1)[1].strip()
		instruction = instruction.split(' so ')[0].strip()
		instruction = instruction.split(' So ')[0].strip()
		instruction = instruction.replace('#', '').strip()

		# Step 2: Split by separators (;, :, ,, then, and)
		instruction = instruction.replace(';', '|SPLIT|')
		instruction = instruction.replace(':', '|SPLIT|')  # Colon separator
		instruction = re.sub(r',\s*then\s+', '|SPLIT|', instruction, flags=re.IGNORECASE)
		instruction = re.sub(r'\s+then\s+', '|SPLIT|', instruction, flags=re.IGNORECASE)
		instruction = re.sub(r'\s+and\s+', '|SPLIT|', instruction, flags=re.IGNORECASE)
		instruction = re.sub(r',\s+', '|SPLIT|', instruction)
		raw_actions = instruction.split('|SPLIT|')

		# Step 3: Parse each action into standard format
		formulated_actions = []
		for raw_action in raw_actions:
			raw_action = raw_action.strip().rstrip('.')
			if not raw_action:
				continue
			standardized = self._standardize_action(raw_action)
			if standardized:
				formulated_actions.append(standardized)

		# Step 4: Post-processing - fill in missing objects for [land_on] and [takeoff_from]
		for i in range(len(formulated_actions)):
			action = formulated_actions[i]
			if action == '[land_on]' and i > 0:
				prev_action = formulated_actions[i-1]
				if prev_action.startswith('[movetowards]'):
					obj_match = re.search(r'<[^>]+>\(\d+\)', prev_action)
					if obj_match:
						formulated_actions[i] = f'[land_on] {obj_match.group(0)}'
			elif action == '[takeoff_from]' and i > 0:
				prev_action = formulated_actions[i-1]
				if prev_action.startswith('[land_on]'):
					obj_match = re.search(r'<[^>]+>\(\d+\)', prev_action)
					if obj_match:
						formulated_actions[i] = f'[takeoff_from] {obj_match.group(0)}'

		return formulated_actions

	def _standardize_action(self, raw_action):
		"""
		Convert natural language action to standard format.
		'I will land on <table>(5)' -> '[land_on] <table>(5)'
		'grab <apple>(9)' -> '[grab] <apple>(9)'
		'putinto <apple>(9) into <basket>(22)' -> '[putinto] <apple>(9) into <basket>(22)'
		"""
		import re
		raw_action = raw_action.strip()
		objects = re.findall(r'<[^>]+>\(\d+\)', raw_action)
		action_lower = raw_action.lower()

		prefixes = ['i will ', 'i\'ll ', 'i ', 'please ', 'hello ']
		for prefix in prefixes:
			if action_lower.startswith(prefix):
				action_lower = action_lower[len(prefix):]
				break

		agent_class = self.agent_node.get('class_name', '').lower()

		if 'quadrotor' in agent_class:
			if 'land' in action_lower:
				return f'[land_on] {objects[0]}' if objects else '[land_on]'
			elif 'takeoff' in action_lower or 'take off' in action_lower:
				return f'[takeoff_from] {objects[0]}' if objects else '[takeoff_from]'
			elif 'movetowards' in action_lower or 'move towards' in action_lower or 'movetoward' in action_lower or 'move toward' in action_lower:
				if objects:
					return f'[movetowards] {objects[0]}'
		elif 'robot dog' in agent_class or 'robot_dog' in agent_class:
			if 'open' in action_lower:
				if objects:
					return f'[open] {objects[0]}'
			elif 'close' in action_lower:
				if objects:
					return f'[close] {objects[0]}'
			elif 'put' in action_lower or 'place' in action_lower or 'release' in action_lower:
				if 'into' in action_lower or 'putin' in action_lower or 'put_into' in action_lower:
					if len(objects) >= 2:
						return f'[putinto] {objects[0]} into {objects[1]}'
					elif len(objects) == 1:
						return f'[putinto] into {objects[0]}'
				elif ' on ' in action_lower or 'puton' in action_lower or 'put_on' in action_lower:
					if len(objects) >= 2:
						return f'[puton] {objects[0]} on {objects[1]}'
					elif len(objects) == 1:
						return f'[puton] on {objects[0]}'
			elif 'grab' in action_lower or 'pick' in action_lower or 'take' in action_lower:
				if 'take off' not in action_lower and 'takeoff' not in action_lower:
					if objects:
						return f'[grab] {objects[0]}'
			elif 'movetowards' in action_lower or 'move towards' in action_lower or 'movetoward' in action_lower or 'move toward' in action_lower or 'go to' in action_lower:
				if objects:
					return f'[movetowards] {objects[0]}'
		elif 'robot arm' in agent_class or 'robot_arm' in agent_class:
			if 'open' in action_lower:
				if objects:
					return f'[open] {objects[0]}'
			elif 'close' in action_lower:
				if objects:
					return f'[close] {objects[0]}'
			elif 'put' in action_lower or 'place' in action_lower or 'release' in action_lower:
				if 'into' in action_lower or 'putin' in action_lower or 'put_into' in action_lower:
					if len(objects) >= 2:
						return f'[putinto] {objects[0]} into {objects[1]}'
					elif len(objects) == 1:
						return f'[putinto] into {objects[0]}'
				elif ' on ' in action_lower or 'puton' in action_lower or 'put_on' in action_lower:
					if len(objects) >= 2:
						return f'[puton] {objects[0]} on {objects[1]}'
					elif len(objects) == 1:
						return f'[puton] on {objects[0]}'
			elif 'grab' in action_lower or 'pick' in action_lower or 'take' in action_lower:
				if 'take off' not in action_lower and 'takeoff' not in action_lower:
					if objects:
						return f'[grab] {objects[0]}'
		return None

	def get_action(self, observation, chat_agent_info, goal, goal_instruction):
		import re
		
		new_instruction_received = chat_agent_info.get('instruction') is not None
		if new_instruction_received:
			self.last_instruction = chat_agent_info.get('instruction')
			self._pending_new_instruction = True

		satisfied, unsatisfied = self.check_progress(observation, goal) 

		self.unsatisfied = unsatisfied
		if len(satisfied) > 0:
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

		# Find objects that have not been reached yet (exclude grabbed, reachable, rooms, agents, floor, high surfaces)
		self.unreached_objects = [
			node for node in obs['nodes']
			if not (
				node == self.grabbed_objects or 
				node in self.reachable_objects or
				node['category'] in ['Rooms', 'Agents', 'Floor'] or
				'HIGH_HEIGHT' in node['properties'] or
				'ON_HIGH_SURFACE' in node['properties']
			)
		]

		# ── Compute live state metadata for context-conditioned cache ──
		agent_class = self.agent_node.get('class_name', '').lower().replace(' ', '_')
		self._cache_agents_on_same_surface = []
		self._cache_agents_in_room = []
		self._cache_agents_on_land_surface = []
		self._cache_basket_contents = []
		self._cache_objects_on_same_surface = []

		my_id = self.agent_node['id']
		my_surface_id = self.on_surfaces['id'] if self.on_surfaces else None
		my_room_id = self.current_room['id'] if self.current_room else None

		# Build id->room and id->surface maps for other agents
		for node in obs['nodes']:
			if node['category'] == 'Agents' and node['id'] != my_id:
				agent_label = f"<{node['class_name']}>({node['id']})"
				# Check if this agent is in same room (for robot_dog)
				for e in obs['edges']:
					if e['from_id'] == node['id']:
						if e['relation_type'] == 'INSIDE' and my_room_id and e['to_id'] == my_room_id:
							self._cache_agents_in_room.append(agent_label)
						if e['relation_type'] == 'ON' and my_surface_id and e['to_id'] == my_surface_id:
							self._cache_agents_on_same_surface.append(agent_label)
							self._cache_agents_on_land_surface.append(agent_label)

		# objects_on_same_surface: for robot_arm, use on_same_surfaces
		if 'robot_arm' in agent_class:
			for obj_node in self.on_same_surfaces:
				obj_label = f"<{obj_node['class_name']}>({obj_node['id']})"
				self._cache_objects_on_same_surface.append(obj_label)

		# quadrotor-specific: basket_contents and agent_surface_ids
		self._cache_agent_surface_ids = set()
		if 'quadrotor' in agent_class:
			# Find basket attached to quadrotor (WITH_QUADROTOR)
			basket_id = None
			for node in obs['nodes']:
				if 'WITH_QUADROTOR' in node.get('properties', []):
					basket_id = node['id']
					break
			if basket_id is not None:
				for e in obs['edges']:
					if e['to_id'] == basket_id and e['relation_type'] == 'INSIDE':
						item_node = self.id2node.get(e['from_id'])
						if item_node:
							self._cache_basket_contents.append(f"<{item_node['class_name']}>({item_node['id']})")

			# Find surface IDs that have other agents ON them (for land_on_agent)
			for node in obs['nodes']:
				if node['category'] == 'Agents' and node['id'] != my_id:
					for e in obs['edges']:
						if e['from_id'] == node['id'] and e['relation_type'] == 'ON':
							self._cache_agent_surface_ids.add(e['to_id'])

		self.doors = []
		self.next_rooms = []
		self.doors = [x for x in obs['nodes'] if x['class_name'] == 'door']
		for door in self.doors:
			for edge in self.init_graph['edges']:
				if edge['relation_type'] == "LEADING TO" and edge['from_id'] == door['id'] and edge['to_id'] != self.current_room["id"]:
						self.next_rooms.append([self.init_id2node[edge['to_id']], door])

		# ── Process instruction queue after observation is ready ──
		if new_instruction_received:
			self.instruction_queue.clear()
			formulated_actions = self.instruction_parser(self.last_instruction)
			print(f"Cache: agent {self.agent_node['class_name']}({self.agent_node['id']})")
			print(f"Cache: formulated_actions: {formulated_actions}")
			self.instruction_queue = formulated_actions

			# Remove actions not in agent's capability
			agent_class_lower = self.agent_node.get('class_name', '').lower()
			capability_filtered = []
			for action in self.instruction_queue:
				action_match = re.match(r'\[(\w+)\]', action)
				if action_match:
					action_type = action_match.group(1)
					if 'quadrotor' in agent_class_lower:
						valid = ['land_on', 'movetowards', 'takeoff_from']
					elif 'robot_dog' in agent_class_lower or 'robot dog' in agent_class_lower:
						valid = ['open', 'close', 'grab', 'putinto', 'puton', 'movetowards']
					elif 'robot_arm' in agent_class_lower or 'robot arm' in agent_class_lower:
						valid = ['open', 'close', 'grab', 'putinto', 'puton']
					else:
						valid = []
					if action_type in valid:
						capability_filtered.append(action)
				else:
					capability_filtered.append(action)
			self.instruction_queue = capability_filtered

			# Quadrotor: insert takeoff if on surface and first action is movetowards
			if 'quadrotor' in agent_class_lower:
				is_flying = self.on_surfaces is None
				if not is_flying and len(self.instruction_queue) > 0:
					first = self.instruction_queue[0]
					if 'movetowards' in first.lower():
						surface_label = f"<{self.on_surfaces['class_name']}>({self.on_surfaces['id']})" if self.on_surfaces else ''
						self.instruction_queue.insert(0, f'[takeoff_from] {surface_label}' if surface_label else '[takeoff_from]')

				# Fill in missing object for any [takeoff_from] using current surface
				if self.on_surfaces:
					surface_label = f"<{self.on_surfaces['class_name']}>({self.on_surfaces['id']})"
					for i, action in enumerate(self.instruction_queue):
						if action == '[takeoff_from]':
							self.instruction_queue[i] = f'[takeoff_from] {surface_label}'

			print(f"Cache: instruction_queue: {self.instruction_queue}")
			# Inject oracle bigrams into cache for oracle-augmented action selection
			self.action_cache.inject_oracle_bigrams(self.last_action, self.instruction_queue, action_history=self.action_history)

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

		# Get available plans and extract action names
		_, _, available_plans_list = self._LLM.get_available_plans(
			self.agent_node, self.next_rooms, self.all_landable_surfaces, 
			self.landable_surfaces, self.on_surfaces, self.grabbed_objects, 
			self.reachable_objects, self.unreached_objects, self.on_same_surfaces
		)

		# print("Agent", self.agent_node['class_name'], "(", self.agent_node['id'], "): available_plans_list", available_plans_list)
		self._current_available_plans = available_plans_list

		# Check if async LLM result is ready
		llm_plan, message, a_info = self.check_async_llm_result()

		# Update cache with LLM's choice using context saved at query time
		if llm_plan is not None:
			self.action_cache.update(llm_plan, self._llm_query_prev_action, self._llm_query_available_actions, new_segment=self._llm_query_new_segment, last_instruction=self._llm_query_last_instruction, grabbed_objects=self._llm_query_grabbed_objects, basket_contents=self._llm_query_basket_contents, agents_on_same_surface=self._llm_query_agents_on_same_surface, agents_in_room=self._llm_query_agents_in_room, agents_on_land_surface=self._llm_query_agents_on_land_surface, unsatisfied=self._llm_query_unsatisfied, objects_on_same_surface=self._llm_query_objects_on_same_surface, agent_surface_ids=self._llm_query_agent_surface_ids, obs=self._llm_query_obs)

		# Submit async LLM query if not already running and we have instruction context
		if (not self.llm_done_flag.value) and self.async_result is None and self.last_instruction:
			self.submit_async_llm_query()

		# Cache access
		plan, message, decision_source = self.action_cache.get_cached_action(available_plans_list, prev_action=self.last_action, goal=goal,\
			llm_plan=llm_plan,unsatisfied=unsatisfied,action_history=self.action_history,obs=chat_agent_info['observation'],\
			grabbed_objects=self.grabbed_objects, agents_on_same_surface=self._cache_agents_on_same_surface,\
			agents_in_room=self._cache_agents_in_room, agents_on_land_surface=self._cache_agents_on_land_surface,\
			basket_contents=self._cache_basket_contents, objects_on_same_surface=self._cache_objects_on_same_surface,\
			agent_surface_ids=self._cache_agent_surface_ids, last_instruction=self.last_instruction)

		# Debug logging: plan selection details
		agent_label = f"<{self.agent_node['class_name']}>({self.agent_node['id']})"
		print(f"  [{agent_label}] step={self.steps} source={decision_source} plan={plan} llm_plan={llm_plan} queue={self.instruction_queue} available={available_plans_list}")

		if message == "I executed [wait]":
			message = None
		# Ensure action_list is always populated in a_info
		if "action_list" not in a_info:
			a_info["action_list"] = available_plans_list
		a_info["decision_source"] = decision_source

		# ── Save per-step metadata for rich JSONL logging ──
		step_meta = {
			"action": plan if plan is not None else "[wait]",
			"decision_source": decision_source,
			"available_actions": list(available_plans_list),
		}
		self.step_metadata[self.steps] = step_meta

		if plan != '[wait]':
			self.last_action = plan
			self.action_history.append(f"{plan} at step {self.steps}")
		else:
			self.last_action = '[wait]'
			plan = None
		a_info.update({"steps": self.steps})
		info.update({"LLM": a_info})

		return plan, message, info

	def __del__(self):
		"""Cleanup the process pool when the object is destroyed"""
		if hasattr(self, 'agent_pool'):
			self.agent_pool.close()
			self.agent_pool.join()
		if hasattr(self, 'manager'):
			self.manager.shutdown()
