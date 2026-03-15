"""
Action Cache for storing and accessing cached action patterns from a unified
cache_data JSON file produced by generate_pefa_cache_data.py.

The JSON contains:
  - selection_rates: {action_type: {chosen, total, rate}}
  - bigrams (unconditioned): {prev_type: [[next_type, count], ...]}
  - conditioned_bigrams: {context_key: {prev_type: [[next_type, count], ...]}}
  - context_features: list of feature names used for this agent type

Context keys encode binary state features:
  c = has_colocated_agent (0/1)
  b = basket_has_items (0/1, quadrotor only)
  g = has_grabbed (0/1)
  s = goal_object_on_same_surface (0/1, robot_arm only)
"""
import os
import re
import json
import random


class ActionCache:
	"""
	Cache class for accessing action patterns based on historical analysis.
	Loads data from a unified cache_data_{agent_type}.json file.
	"""
	
	def __init__(self, agent_type, base_dir=None):
		"""
		Initialize the cache for a specific agent type.
		
		Args:
			agent_type (str): Agent type (e.g., 'quadrotor', 'robot_arm', 'robot_dog')
			base_dir (str): Base directory containing the cache JSON files.
						   If None, uses the same directory as this file.
		"""
		self.agent_type = agent_type
		
		if base_dir is None:
			base_dir = os.path.dirname(os.path.abspath(__file__))
		
		self.cache_file = os.path.join(base_dir, f"cache_data_{agent_type}.json")
		
		# Core data loaded from JSON
		self.conditioned_bigrams = {}    # {ctx_key: {prev_type: {next_type: count}}}
		self.full_bigrams = {}           # {prev_type: {next_type: count}} (unconditioned fallback)
		self.selection_rates = {}        # {action_type: rate}
		self.selection_counts = {}       # {action_type: [selected, total]} for dynamic updates
		self.context_features = []       # list of feature names
		self._pending_new_segment = False
		self._oracle_chain = []      # ordered list of concrete actions from oracle
		self._oracle_chain_idx = 0   # index of next action to suggest
		
		self._load_cache_json()
	
	def _load_cache_json(self):
		"""Load all cache data from the unified JSON file."""
		try:
			with open(self.cache_file, 'r') as f:
				data = json.load(f)
			
			# Selection rates
			for atype, stats in data.get('selection_rates', {}).items():
				self.selection_rates[atype] = stats.get('rate', 0.0)
				self.selection_counts[atype] = [stats.get('chosen', 0), stats.get('total', 0)]
			
			# Conditioned bigrams: {ctx_key: {prev: [[next, count], ...]}}
			for ctx_key, grouped in data.get('conditioned_bigrams', {}).items():
				self.conditioned_bigrams[ctx_key] = {}
				for prev_type, nexts in grouped.items():
					p = None if prev_type == 'None' else prev_type
					self.conditioned_bigrams[ctx_key][p] = {}
					for next_type, count in nexts:
						self.conditioned_bigrams[ctx_key][p][next_type] = count
			
			# Full (unconditioned) bigrams: {prev: [[next, count], ...]}
			for prev_type, nexts in data.get('bigrams', {}).items():
				p = None if prev_type == 'None' else prev_type
				self.full_bigrams[p] = {}
				for next_type, count in nexts:
					self.full_bigrams[p][next_type] = count
			
			# Context features list
			self.context_features = data.get('context_features', [])
			
		except FileNotFoundError:
			print(f"Warning: Cache file not found: {self.cache_file}")
		except Exception as e:
			print(f"Error loading cache JSON: {e}")
	
	# ── Context key computation from live state ──────────────────────────────
	
	def _compute_context_key(self, grabbed_objects=None, agents_on_same_surface=None,
							  agents_in_room=None, agents_on_land_surface=None,
							  basket_contents=None, unsatisfied=None,
							  objects_on_same_surface=None):
		"""
		Compute context key from live agent state, matching the features
		used during cache generation.
		
		Returns:
			str: Context key like 'c1_g0' or 'b1_c0_g0_s1'
		"""
		ctx = {}
		
		# has_colocated_agent
		if self.agent_type == 'robot_arm':
			colocated = agents_on_same_surface or []
			ctx['has_colocated_agent'] = 1 if len(colocated) > 0 else 0
		elif self.agent_type == 'robot_dog':
			colocated = agents_in_room or []
			ctx['has_colocated_agent'] = 1 if len(colocated) > 0 else 0
		elif self.agent_type == 'quadrotor':
			colocated = agents_on_land_surface or []
			ctx['has_colocated_agent'] = 1 if len(colocated) > 0 else 0
		else:
			ctx['has_colocated_agent'] = 0
		
		# basket_has_items (quadrotor only)
		if self.agent_type == 'quadrotor':
			basket = basket_contents or []
			ctx['basket_has_items'] = 1 if len(basket) > 0 else 0
		
		# has_grabbed (not for quadrotor)
		if self.agent_type != 'quadrotor':
			ctx['has_grabbed'] = 1 if grabbed_objects else 0
		
		# goal_object_on_same_surface (robot_arm only)
		if self.agent_type == 'robot_arm':
			goal_objects = set()
			if unsatisfied:
				for key in unsatisfied.keys():
					for obj_match in re.finditer(r'<[^>]+>\(\d+\)', key):
						goal_objects.add(obj_match.group(0))
			surface_objs = set(objects_on_same_surface or [])
			ctx['goal_object_on_same_surface'] = 1 if goal_objects & surface_objs else 0
		
		# Build key string in sorted order
		parts = []
		for k in sorted(ctx.keys()):
			if k == 'has_colocated_agent':
				short = 'c'
			elif k == 'basket_has_items':
				short = 'b'
			elif k == 'has_grabbed':
				short = 'g'
			elif k == 'goal_object_on_same_surface':
				short = 's'
			else:
				short = k[0]
			parts.append(f"{short}{ctx[k]}")
		return '_'.join(parts)
	
	@staticmethod
	def _extract_primary_target(action_str):
		"""Extract first <object>(id) from action string for target continuity.
		
		'[movetowards] <door>(7)'                    -> '<door>(7)'
		'[putinto] <bread>(26) into <basket>(25)'     -> '<bread>(26)'
		"""
		if not action_str:
			return None
		m = re.search(r'<[^>]+>\(\d+\)', action_str)
		return m.group(0) if m else None

	def _get_candidate_counts(self, prev_action_type, ctx_key):
		"""
		Get bigram counts for (prev_action_type -> next).
		Tries conditioned bigrams first, falls back to full bigrams.
		
		Returns:
			dict: {next_action_type: count}
		"""
		# Try conditioned bigrams first
		if ctx_key in self.conditioned_bigrams:
			cond_table = self.conditioned_bigrams[ctx_key]
			if prev_action_type in cond_table:
				return cond_table[prev_action_type]
		
		# Fallback to unconditioned (full) bigrams
		if prev_action_type in self.full_bigrams:
			return self.full_bigrams[prev_action_type]
		
		return {}

	@staticmethod
	def _base_verb(action_type):
		"""Extract base verb from subdivided action type.
		'movetowards_goal_obj' -> 'movetowards'
		'grab_goal_subj'       -> 'grab'
		'land_on_agent_surface' -> 'land_on'
		"""
		if not action_type:
			return action_type
		# Known multi-word verbs
		for prefix in ('takeoff_from', 'land_on', 'movetowards'):
			if action_type.startswith(prefix):
				return prefix
		# Single-word: grab, putinto, puton, open, close, wait
		return action_type.split('_')[0]
	
	def _dedup_match(self, chain_action, executed_action):
		"""
		Match for dedup purposes. More lenient than _actions_match:
		  - takeoff_from: type-only match (surface is incidental, not a goal target)
		  - Others: delegate to _actions_match
		"""
		a = re.match(r'\[(\w+)\]', chain_action)
		b = re.match(r'\[(\w+)\]', executed_action)
		if not a or not b:
			return False
		if a.group(1) == 'takeoff_from' and b.group(1) == 'takeoff_from':
			return True
		return self._actions_match(chain_action, executed_action)

	def _actions_match(self, parsed_action, available_action):
		"""
		Check if a parsed action matches an available action.

		Args:
			parsed_action (str): Action from instruction_parser, e.g., '[grab] <apple>(9)'
			available_action (str): Action from available_actions list

		Returns:
			bool: True if actions match (same type and objects)
		"""
		# Extract action type
		parsed_type = re.match(r'\[(\w+)\]', parsed_action)
		available_type = re.match(r'\[(\w+)\]', available_action)

		if not parsed_type or not available_type:
			return False

		# Check if action types match
		action_type = parsed_type.group(1)
		if action_type != available_type.group(1):
			return False

		# Extract all objects from both actions
		parsed_objects = re.findall(r'<[^>]+>\(\d+\)', parsed_action)
		available_objects = re.findall(r'<[^>]+>\(\d+\)', available_action)

		# Special handling for land_on and takeoff_from - they can be executed without objects
		if action_type in ['land_on', 'takeoff_from']:
			# If parsed action has no object, match any available action of same type
			if len(parsed_objects) == 0:
				return True
			# If parsed action has object, must match exactly
			elif len(available_objects) > 0:
				return parsed_objects[0] == available_objects[0]
			else:
				return False

		# For other actions, standard matching
		# For actions without objects (like [wait])
		if len(parsed_objects) == 0 and len(available_objects) == 0:
			return True

		# Check if all objects match
		if len(parsed_objects) != len(available_objects):
			return False

		# Compare objects (they should be in the same order)
		for p_obj, a_obj in zip(parsed_objects, available_objects):
			if p_obj != a_obj:
				return False

		return True

	def inject_oracle_bigrams(self, prev_action, oracle_actions, action_history=None):
		"""
		Inject temporary ordered action chain from oracle instruction.
		Trims prefix already executed in recent history before registering.
		
		Args:
			prev_action: The action executed just before oracle instruction arrived
			oracle_actions: List of concrete actions parsed from oracle instruction
			action_history: List of 'action at step N' strings from agent history
		"""
		if not oracle_actions:
			self._oracle_chain = []
			self._oracle_chain_idx = 0
			return

		# Parse history into action strings
		past_actions = []
		if action_history:
			for entry in action_history:
				if " at step " in entry:
					past_actions.append(entry.split(" at step ")[0])

		# Trim already-executed prefix: only if history TAIL matches chain start
		# (suffix-based: avoids trimming stale movetowards that agent moved away from)
		start_idx = 0
		if past_actions:
			max_prefix = min(len(oracle_actions), len(past_actions))
			for prefix_len in range(max_prefix, 0, -1):
				tail = past_actions[-prefix_len:]
				all_match = all(
					self._dedup_match(oracle_actions[j], tail[j])
					for j in range(prefix_len)
				)
				if all_match:
					start_idx = prefix_len
					break

		self._oracle_chain = list(oracle_actions[start_idx:])
		self._oracle_chain_idx = 0

	def get_cached_action(self, available_actions, prev_action=None, goal=None,llm_plan=None,\
		unsatisfied=None,action_history=None,obs=None,\
		grabbed_objects=None, agents_on_same_surface=None, agents_in_room=None,\
		agents_on_land_surface=None, basket_contents=None, objects_on_same_surface=None,\
		agent_surface_ids=None, last_instruction=None):
		"""
		Get cached action based on available actions, previous action, and live state.
		
		Priority:
		  0. Oracle-augmented cache (concrete actions from oracle instruction)
		  1. LLM plan (not in history -> override cache unconditionally)
		  2. 100% selection rate action from bigram -> immediate
		  3. If bigram top candidate is _instruction type -> wait for LLM
		  4. Best scored action from cache after state filtering
		  5. [wait]
		
		Returns:
		  (action, message, source) where source is one of:
		    'cache_oracle_augmented', 'llm', 'cache_100%', '_instruction_wait', 'cache_bigram', 'wait'
		"""

		# ── Anti-oscillation: extract recent actions early ──
		WINDOW_SIZE = 6
		MAX_REPEATS = 2
		past_actions = []
		if action_history and len(action_history) > 0:
			for history_entry in action_history:
				if " at step " in history_entry:
					action_part = history_entry.split(" at step ")[0]
					past_actions.append(action_part)
		recent_actions = past_actions[-WINDOW_SIZE:]

		def _action_blocked(action):
			"""Return True if action would cause oscillation or exceeded repeat limit."""
			if recent_actions.count(action) >= MAX_REPEATS:
				return True

			action_match = re.match(r'\[(\w+)\]', action)
			if not action_match:
				return False
			at = action_match.group(1)

			# A→B→A oscillation detection for land_on / takeoff_from
			if len(recent_actions) >= 2 and action == recent_actions[-2] and action != recent_actions[-1]:
				if at in ('land_on', 'takeoff_from'):
					return True

			# Position-based: block movetowards only if agent is already at the destination
			if at == 'movetowards':
				proposed_dest = re.findall(r'<[^>]+>\(\d+\)', action)
				if proposed_dest:
					for past in reversed(recent_actions):
						pm = re.match(r'\[(\w+)\]', past)
						if pm and pm.group(1) == 'movetowards':
							past_dest = re.findall(r'<[^>]+>\(\d+\)', past)
							if past_dest and past_dest[0] == proposed_dest[0]:
								return True  # already at this destination
							break

			return False

		# ── Precompute goal destinations per subject from ALL goals (not just unsatisfied) ──
		# Using full goal ensures dedup still works after a goal is satisfied,
		# preventing the agent from re-grabbing correctly placed objects.
		goal_destinations = {}  # {subject_str: set of destination_strs}
		goal_source = goal if goal else unsatisfied
		if goal_source:
			for key in goal_source.keys():
				key_match = re.match(r'(.+?)_(<[^>]+>\(\d+\))_(<[^>]+>\(\d+\))', key)
				if key_match:
					subj = key_match.group(2)
					dest = key_match.group(3)
					if subj not in goal_destinations:
						goal_destinations[subj] = set()
					goal_destinations[subj].add(dest)

		# ── Filter out grab for objects already grabbed-and-placed at correct goal destination ──
		placed_objects = set()
		for i, past in enumerate(past_actions):
			pm = re.match(r'\[(\w+)\]', past)
			if pm and pm.group(1) == 'grab':
				po = re.findall(r'<[^>]+>\(\d+\)', past)
				if po:
					obj = po[0]
					for later in past_actions[i + 1:]:
						lm = re.match(r'\[(\w+)\]', later)
						if lm and lm.group(1) in ('putinto', 'puton'):
							lo = re.findall(r'<[^>]+>\(\d+\)', later)
							if lo and lo[0] == obj:
								place_dest = lo[1] if len(lo) > 1 else None
								obj_goals = goal_destinations.get(obj, set())
								if place_dest and place_dest in obj_goals:
									placed_objects.add(obj)
								break
		if placed_objects:
			available_actions = [
				a for a in available_actions
				if not (re.match(r'\[grab\]', a) and
						any(obj in a for obj in placed_objects))
			]

		# ── Advance oracle chain pointer if agent already did the next expected action ──
		# (regardless of decision source: cache_bigram, LLM, etc.)
		if (self._oracle_chain and self._oracle_chain_idx < len(self._oracle_chain)
			and prev_action and self._dedup_match(self._oracle_chain[self._oracle_chain_idx], prev_action)):
			self._oracle_chain_idx += 1

		# ── Step 0: Oracle-augmented cache (sequential, unified dedup) ──
		while self._oracle_chain and self._oracle_chain_idx < len(self._oracle_chain):
			oracle_next = self._oracle_chain[self._oracle_chain_idx]
			oracle_match = re.match(r'\[(\w+)\]', oracle_next)
			if not oracle_match:
				break
			oracle_type = oracle_match.group(1)

			# Unified dedup: skip actions already completed
			skip = False
			if oracle_type in ('takeoff_from', 'land_on'):
				# State-based: infer current state from available actions
				can_do = any(a.startswith(f'[{oracle_type}]') for a in available_actions)
				if not can_do:
					skip = True  # already flying (takeoff) or already landed (land_on)
			elif oracle_type == 'movetowards':
				pass  # always duplicable, never skip
			elif oracle_type == 'grab':
				oracle_objects = re.findall(r'<[^>]+>\(\d+\)', oracle_next)
				# Skip if currently holding that exact object
				if grabbed_objects and oracle_objects:
					obj_id_match = re.search(r'\((\d+)\)', oracle_objects[0])
					if obj_id_match and grabbed_objects['id'] == int(obj_id_match.group(1)):
						skip = True
				# Skip if previously grabbed AND placed at correct goal destination
				if not skip and oracle_objects:
					last_grab_idx = -1
					for i, past in enumerate(past_actions):
						pm = re.match(r'\[(\w+)\]', past)
						if pm and pm.group(1) == 'grab':
							po = re.findall(r'<[^>]+>\(\d+\)', past)
							if po and po[0] == oracle_objects[0]:
								last_grab_idx = i
					if last_grab_idx >= 0:
						for past in past_actions[last_grab_idx + 1:]:
							pm = re.match(r'\[(\w+)\]', past)
							if pm and pm.group(1) in ('putinto', 'puton'):
								po = re.findall(r'<[^>]+>\(\d+\)', past)
								if po and po[0] == oracle_objects[0]:
									place_dest = po[1] if len(po) > 1 else None
									obj_goals = goal_destinations.get(oracle_objects[0], set())
									if place_dest and place_dest in obj_goals:
										skip = True
									break
			else:
				# Other actions (putinto, puton, open, close): check full history
				oracle_objects = re.findall(r'<[^>]+>\(\d+\)', oracle_next)
				for past in past_actions:
					past_match = re.match(r'\[(\w+)\]', past)
					if past_match and past_match.group(1) == oracle_type:
						past_objects = re.findall(r'<[^>]+>\(\d+\)', past)
						if oracle_objects == past_objects:
							skip = True
							break

			if skip:
				self._oracle_chain_idx += 1
				continue

			# Try to match against available actions (movetowards exempt from anti-oscillation)
			for action in available_actions:
				if self._actions_match(oracle_next, action):
					if oracle_type not in ('movetowards', 'land_on', 'takeoff_from') and _action_blocked(action):
						break
					self._oracle_chain_idx += 1
					return action, f"I executed {action}", 'cache_oracle_augmented'
			# oracle_next not in available — skip movetowards waypoints, break on others
			if oracle_type == 'movetowards':
				self._oracle_chain_idx += 1
				continue
			# Fallback for puton/putinto: if exact destination not available,
			# substitute with any available same-type action for the same held object.
			# Handles "put down to open door" when oracle suggests unavailable surface.
			if oracle_type in ('puton', 'putinto'):
				oracle_objs = re.findall(r'<[^>]+>\(\d+\)', oracle_next)
				if oracle_objs:
					held_obj = oracle_objs[0]
					exact_type_matches = []
					alternate_put_matches = []
					for action in available_actions:
						am = re.match(r'\[(\w+)\]', action)
						if am and am.group(1) in ('puton', 'putinto'):
							action_objs = re.findall(r'<[^>]+>\(\d+\)', action)
							if action_objs and action_objs[0] == held_obj:
								if am.group(1) == oracle_type:
									exact_type_matches.append(action)
								else:
									alternate_put_matches.append(action)
					for action in exact_type_matches + alternate_put_matches:
						if not _action_blocked(action):
							self._oracle_chain_idx += 1
							return action, f"I executed {action}", 'cache_oracle_augmented'
			break

		# Clear oracle chain once fully consumed so Step 6.5 _instruction_wait works
		if self._oracle_chain and self._oracle_chain_idx >= len(self._oracle_chain):
			self._oracle_chain = []
			self._oracle_chain_idx = 0

		# ── Step 1: Compute context key from live state ──
		ctx_key = self._compute_context_key(
			grabbed_objects=grabbed_objects,
			agents_on_same_surface=agents_on_same_surface,
			agents_in_room=agents_in_room,
			agents_on_land_surface=agents_on_land_surface,
			basket_contents=basket_contents,
			unsatisfied=unsatisfied,
			objects_on_same_surface=objects_on_same_surface,
		)

		# ── Compute goal_subjects/goal_objects for subdivided type detection ──
		goal_subjects = set()
		goal_objects = set()
		if unsatisfied:
			for key in unsatisfied.keys():
				key_match = re.match(r'(.+?)_(<[^>]+>\(\d+\))_(<[^>]+>\(\d+\))', key)
				if key_match:
					goal_subjects.add(key_match.group(2))  # subject
					goal_objects.add(key_match.group(3))   # object

		# ── Extract drone basket ID from obs for put_drone_basket detection ──
		drone_basket_id = None
		if obs:
			for line in obs.split('\n'):
				if 'WITH_QUADROTOR' in line:
					id_match = re.search(r'<[^>]+>\((\d+)\)', line)
					if id_match:
						drone_basket_id = int(id_match.group(1))
					break

		# ── Step 2: Get bigram counts (conditioned -> unconditioned fallback) ──
		if prev_action is None:
			prev_action_type = None
		else:
			prev_action_type = self._extract_action_type(prev_action, goal_subjects=goal_subjects, goal_objects=goal_objects, agent_surface_ids=agent_surface_ids, drone_basket_id=drone_basket_id, last_instruction=last_instruction, grabbed_objects=grabbed_objects, basket_contents=basket_contents)
			if prev_action_type == 'wait' and not past_actions:
				prev_action_type = None

		candidate_counts = self._get_candidate_counts(prev_action_type, ctx_key)

		# Fallback: if wait bigram returned nothing, try last non-wait action
		if not candidate_counts and prev_action_type == 'wait' and past_actions:
			for past in reversed(past_actions):
				if not re.match(r'\[wait\]', past):
					prev_action_type = self._extract_action_type(past, goal_subjects=goal_subjects, goal_objects=goal_objects, agent_surface_ids=agent_surface_ids, drone_basket_id=drone_basket_id, last_instruction=last_instruction, grabbed_objects=grabbed_objects, basket_contents=basket_contents)
					candidate_counts = self._get_candidate_counts(prev_action_type, ctx_key)
					break

		# ── Step 3: Score available actions = bigram_count * selection_rate ──
		scored_actions = []
		action_type_map = {}  # action_str -> subdivided action type
		for available_action in available_actions:
			action_type = self._extract_action_type(available_action, goal_subjects=goal_subjects, goal_objects=goal_objects, agent_surface_ids=agent_surface_ids, drone_basket_id=drone_basket_id, last_instruction=last_instruction, grabbed_objects=grabbed_objects, basket_contents=basket_contents)
			if action_type:
				action_type_map[available_action] = action_type
				if action_type == 'wait':
					scored_actions.append((available_action, 0))
				else:
					# Lookup priority: exact type → base-verb aggregate
					count = None
					lookup_type = None
					if action_type in candidate_counts:
						count = candidate_counts[action_type]
						lookup_type = action_type
					else:
						# Base-verb fallback: sum candidates sharing same verb
						verb = self._base_verb(action_type)
						agg = sum(c for k, c in candidate_counts.items()
								  if self._base_verb(k) == verb)
						if agg > 0:
							count = agg
							lookup_type = verb
					
					if count is not None:
						# Rate lookup: exact type → base-verb max rate
						rate = self.selection_rates.get(action_type, 0.0)
						if rate == 0.0 and lookup_type != action_type:
							verb = self._base_verb(action_type)
							rate = max((r for k, r in self.selection_rates.items()
										if self._base_verb(k) == verb), default=0.0)
						score = count * rate
						scored_actions.append((available_action, score))
	
		# Sort by score descending, [wait] at end
		wait_actions = [(a, s) for a, s in scored_actions if a == '[wait]']
		other_actions = [(a, s) for a, s in scored_actions if a != '[wait]']
		other_actions.sort(key=lambda x: x[1], reverse=True)
		scored_actions = other_actions + wait_actions
		
		# ── Step 4: State-based filtering ──
		goal_pairs = []
		subjects = []
		objects = []
		
		if unsatisfied:
			for key in unsatisfied.keys():
				key_match = re.match(r'(.+?)_(<[^>]+>\(\d+\))_(<[^>]+>\(\d+\))', key)
				
				if key_match:
					relation = key_match.group(1)
					subject = key_match.group(2)
					obj = key_match.group(3)
					
					goal_pairs.append((relation, subject, obj))
					subjects.append(subject)
					objects.append(obj)
		
		# Check if basket is with quadrotor (has WITH_QUADROTOR property)
		basket_with_quadrotor = False
		basket_obj_full = None
		if obs:
			lines = obs.split('\n')
			for line in lines:
				if 'WITH_QUADROTOR' in line:
					obj_match = re.search(r'<([^>]+)>\((\d+)\)', line)
					if obj_match:
						obj_name = obj_match.group(1)
						obj_id = obj_match.group(2)
						basket_obj_full = f'<{obj_name}>({obj_id})'
						basket_with_quadrotor = True
						break
		
		# Filter scored_actions based on unsatisfied goals
		if subjects or objects:
			filtered_actions = []
			putinto_goal_matched = []
			put_basket_matched = []
			puton_matched = []
			movetowards_goal_matched = []
			movetowards_door_matched = []
			movetowards_basket_matched = []
			
			for action, score in scored_actions:
				action_match = re.match(r'\[(\w+)\]', action)
				if not action_match:
					continue
				action_type = action_match.group(1)
				
				# Skip if movetowards action is blocked by repeat limit
				if action_type == 'movetowards' and _action_blocked(action):
						continue
				
				# Skip actions with zero score except [wait]
				if score == 0 and action_type != 'wait':
					continue
				
				if action_type == 'grab':
					if _action_blocked(action):
						continue
					obj_match = re.search(r'<(.+?)>\((\d+)\)', action)
					if obj_match:
						obj_full = obj_match.group(0)
						if obj_full in subjects:
							filtered_actions.append((action, score))
				
				elif action_type == 'putinto':
					parts = action.split(' into ')
					if len(parts) == 2:
						subject_match = re.search(r'<(.+?)>\((\d+)\)', parts[0])
						container_match = re.search(r'<(.+?)>\((\d+)\)', parts[1])
						if subject_match and container_match:
							subject_full = subject_match.group(0)
							container_full = container_match.group(0)
							
							is_goal = False
							for relation, goal_subj, goal_obj in goal_pairs:
								if relation == 'inside' and goal_subj == subject_full and goal_obj == container_full:
									is_goal = True
									break
							
							if is_goal:
								if _action_blocked(action):
									continue
								putinto_goal_matched.append((action, score))
							elif basket_obj_full and container_full == basket_obj_full:
								put_basket_matched.append((action, score))
				
				elif action_type == 'puton':
					if _action_blocked(action):
						continue
					parts = action.split(' on ')
					if len(parts) == 2:
						subject_match = re.search(r'<(.+?)>\((\d+)\)', parts[0])
						surface_match = re.search(r'<(.+?)>\((\d+)\)', parts[1])
						if subject_match and surface_match:
							subject_full = subject_match.group(0)
							surface_full = surface_match.group(0)
							is_corresponding = False
							for relation, goal_subj, goal_obj in goal_pairs:
								if relation == 'on' and goal_subj == subject_full and goal_obj == surface_full:
									is_corresponding = True
									break
							
							if is_corresponding:
								puton_matched.append((action, score))
				
				elif action_type == 'movetowards':
					obj_match = re.search(r'<(.+?)>\((\d+)\)', action)
					if obj_match:
						obj_full = obj_match.group(0)
						if obj_full in subjects or obj_full in objects:
							movetowards_goal_matched.append((action, score))
						elif action_type_map.get(action) == 'movetowards_door':
							movetowards_door_matched.append((action, score))
						elif basket_obj_full and obj_full == basket_obj_full:
							movetowards_basket_matched.append((action, score))
				
				elif action_type == 'land_on':
					if _action_blocked(action):
						continue
					surface_match = re.search(r'<(.+?)>\((\d+)\)', action)
					if surface_match:
						surface_full = surface_match.group(0)
						if surface_full in subjects or surface_full in objects:
							filtered_actions.append((action, score))
				
				else:
					filtered_actions.append((action, score))
			
			# Shuffle within each group for randomness
			random.shuffle(putinto_goal_matched)
			random.shuffle(put_basket_matched)
			random.shuffle(puton_matched)
			random.shuffle(movetowards_goal_matched)
			random.shuffle(movetowards_door_matched)
			random.shuffle(movetowards_basket_matched)
			
			all_putinto = putinto_goal_matched + put_basket_matched
			all_puton = puton_matched
			all_movetowards = movetowards_goal_matched + movetowards_door_matched + movetowards_basket_matched
			
			action_type_groups = {}
			for action, score in filtered_actions:
				action_match = re.match(r'\[(\w+)\]', action)
				if action_match:
					action_type = action_match.group(1)
					if action_type not in action_type_groups:
						action_type_groups[action_type] = []
					action_type_groups[action_type].append((action, score))
			
			for action_type in action_type_groups:
				random.shuffle(action_type_groups[action_type])
			
			filtered_actions_randomized = []
			for action_type, actions in action_type_groups.items():
				filtered_actions_randomized.extend(actions)
			
			if all_putinto:
				max_putinto_score = all_putinto[0][1] if all_putinto else 0
				insert_idx = 0
				for i, (act, sc) in enumerate(filtered_actions_randomized):
					if sc < max_putinto_score:
						break
					insert_idx = i + 1
				filtered_actions_randomized = filtered_actions_randomized[:insert_idx] + all_putinto + filtered_actions_randomized[insert_idx:]
			
			if all_puton:
				max_puton_score = all_puton[0][1] if all_puton else 0
				insert_idx = 0
				for i, (act, sc) in enumerate(filtered_actions_randomized):
					if sc < max_puton_score:
						break
					insert_idx = i + 1
				filtered_actions_randomized = filtered_actions_randomized[:insert_idx] + all_puton + filtered_actions_randomized[insert_idx:]
			
			if all_movetowards:
				max_movetowards_score = all_movetowards[0][1] if all_movetowards else 0
				insert_idx = 0
				for i, (act, sc) in enumerate(filtered_actions_randomized):
					if sc < max_movetowards_score:
						break
					insert_idx = i + 1
				filtered_actions_randomized = filtered_actions_randomized[:insert_idx] + all_movetowards + filtered_actions_randomized[insert_idx:]
			
			scored_actions = filtered_actions_randomized

		# ── Step 5: LLM plan (not in history → override cache unconditionally) ──
		if llm_plan and llm_plan in available_actions and not _action_blocked(llm_plan):
			messege = f"I executed {llm_plan}" if llm_plan != '[wait]' else None
			return llm_plan, messege, 'llm'

		# ── Step 6: 100% rate fast-track ──
		for action, score in scored_actions:
			if action == '[wait]':
				continue
			atype = action_type_map.get(action)
			if atype:
				rate = self.selection_rates.get(atype, 0.0)
				if rate >= 1.0 and '_instruction' not in atype and not _action_blocked(action):
					return action, f"I executed {action}", 'cache_100%'

		# ── Step 6.5: If bigram top candidate is _instruction type, wait for LLM ──
		# Only wait if no oracle instruction has been received yet;
		# once oracle chain was processed, fall through to cache_bigram
		if not self._oracle_chain:
			if candidate_counts:
				top_type = max(candidate_counts, key=candidate_counts.get)
				if '_instruction' in top_type:
					return '[wait]', None, '_instruction_wait'

		# ── Step 7: Return best scored action or wait ──
		for best_action, _score in scored_actions:
			if not _action_blocked(best_action):
				messege = f"I executed {best_action}" if best_action != '[wait]' else None
				return best_action, messege, 'cache_bigram'

		return '[wait]', None, 'wait'
	
	def _extract_action_type(self, action, goal_subjects=None, goal_objects=None, agent_surface_ids=None, drone_basket_id=None, last_instruction=None, grabbed_objects=None, basket_contents=None):
		"""
		Extract action type from formatted action string, subdividing by target context.
		
		Args:
			action: Full action string
			goal_subjects: Set of goal subject strings like {'<bread>(26)'}
			goal_objects: Set of goal object strings like {'<fridge>(14)'}
			agent_surface_ids: Set of surface IDs (int) that have another agent ON them
			drone_basket_id: ID (int) of the quadrotor's basket, if known
			last_instruction: Oracle instruction string (for _instruction suffix detection)
			grabbed_objects: Grabbed object dict (truthy if agent is holding something)
			basket_contents: List of basket contents (quadrotor only)
		"""
		if action is None:
			return None
		match = re.match(r'\[(\w+)\]', action)
		if not match:
			return None
		base_type = match.group(1)

		if base_type == 'movetowards':
			target_match = re.search(r'<([^>]+)>\((\d+)\)', action)
			if target_match:
				target_name = target_match.group(1)
				target_id = int(target_match.group(2))
				target_full = target_match.group(0)
				if drone_basket_id is not None and target_id == drone_basket_id:
					return 'movetowards_drone_basket'
				if agent_surface_ids and target_id in agent_surface_ids:
					return 'movetowards_agent_surface'
				if goal_subjects and target_full in goal_subjects:
					return 'movetowards_goal_subj'
				if goal_objects and target_full in goal_objects:
					if grabbed_objects:
						return 'movetowards_goal_obj'
					else:
						return 'movetowards_goal_obj_open'
				if 'door' in target_name.lower():
					return 'movetowards_door'
				instr = last_instruction or ''
				if target_full in instr:
					return 'movetowards_instruction'
			return 'movetowards'

		if base_type == 'grab':
			obj_match = re.search(r'<[^>]+>\(\d+\)', action)
			if obj_match and goal_subjects and obj_match.group(0) in goal_subjects:
				return 'grab_goal_subj'
			return 'grab'

		if base_type == 'land_on':
			surface_match = re.search(r'<[^>]+>\((\d+)\)', action)
			if surface_match:
				surface_id = int(surface_match.group(1))
				if agent_surface_ids and surface_id in agent_surface_ids:
					return 'land_on_agent_surface'
				full_match = re.search(r'<[^>]+>\(\d+\)', action)
				if full_match:
					instr = last_instruction or ''
					if full_match.group(0) in instr:
						return 'land_on_instruction'
			return 'land_on'

		if base_type == 'putinto':
			target_match = re.search(r'into <[^>]+>\((\d+)\)', action)
			if target_match and 'basket' in action.lower():
				basket_id = int(target_match.group(1))
				if drone_basket_id is not None and basket_id == drone_basket_id:
					return 'put_drone_basket'
				return 'put_basket'
			# Check if destination is a goal object
			dest_match = re.search(r'into (<[^>]+>\(\d+\))', action)
			if dest_match and goal_objects and dest_match.group(1) in goal_objects:
				return 'put_goal_obj'
			return 'putinto'

		if base_type == 'puton':
			# Check if destination is a goal object
			dest_match = re.search(r'on (<[^>]+>\(\d+\))', action)
			if dest_match and goal_objects and dest_match.group(1) in goal_objects:
				return 'put_goal_obj'
			return 'puton'

		if base_type == 'takeoff_from':
			instr = last_instruction or ''
			if 'takeoff' in instr.lower():
				return 'takeoff_from_instruction'
			return 'takeoff_from'

		return base_type

	def update(self, selected_action, prev_action, available_actions, new_segment=False,
			   last_instruction=None, grabbed_objects=None, basket_contents=None,
			   agents_on_same_surface=None, agents_in_room=None, agents_on_land_surface=None,
			   unsatisfied=None, objects_on_same_surface=None, agent_surface_ids=None, obs=None):
		"""
		Dynamically update cache statistics based on the action taken.
		Updates both selection rates and bigram counts (unconditioned + conditioned).
		
		Args:
			selected_action (str): The action that was selected (e.g., '[grab] <bread>(26)')
			prev_action (str): The previous action (None if first action)
			available_actions (list): List of all available actions at this step
			new_segment (bool): True if a new instruction/LLM request started this segment
			last_instruction: Oracle instruction string
			grabbed_objects: Grabbed object dict (truthy if agent is holding something)
			basket_contents: List of basket contents (quadrotor only)
			agents_on_same_surface: List of colocated agent labels (robot_arm)
			agents_in_room: List of agents in same room (robot_dog)
			agents_on_land_surface: List of agents on same land surface (quadrotor)
			unsatisfied: Dict of unsatisfied goals
			objects_on_same_surface: List of object labels on same surface (robot_arm)
			agent_surface_ids: Set of surface IDs with other agents
			obs: Observation text (for drone_basket_id extraction)
		"""
		# ── Derive goal context for accurate type detection ──
		goal_subjects = set()
		goal_objects = set()
		if unsatisfied:
			for key in unsatisfied.keys():
				key_match = re.match(r'(.+?)_(<[^>]+>\(\d+\))_(<[^>]+>\(\d+\))', key)
				if key_match:
					goal_subjects.add(key_match.group(2))
					goal_objects.add(key_match.group(3))

		drone_basket_id = None
		if obs:
			for line in obs.split('\n'):
				if 'WITH_QUADROTOR' in line:
					id_match = re.search(r'<[^>]+>\((\d+)\)', line)
					if id_match:
						drone_basket_id = int(id_match.group(1))
					break

		type_kwargs = dict(goal_subjects=goal_subjects, goal_objects=goal_objects,
						   agent_surface_ids=agent_surface_ids, drone_basket_id=drone_basket_id,
						   last_instruction=last_instruction, grabbed_objects=grabbed_objects,
						   basket_contents=basket_contents)

		selected_type = self._extract_action_type(selected_action, **type_kwargs)
		is_wait = (selected_type is None or selected_type == 'wait')

		# ── Update selection rates (numerator/denominator) ──
		available_types = set()
		for action in available_actions:
			atype = self._extract_action_type(action, **type_kwargs)
			if atype and atype != 'wait':
				available_types.add(atype)

		for atype in available_types:
			if atype not in self.selection_counts:
				self.selection_counts[atype] = [0, 0]
			self.selection_counts[atype][1] += 1

		if not is_wait and selected_type in self.selection_counts:
			self.selection_counts[selected_type][0] += 1

		for atype, (sel, tot) in self.selection_counts.items():
			self.selection_rates[atype] = sel / tot if tot > 0 else 0.0

		# Mark new segment start
		if new_segment:
			self._pending_new_segment = True

		if is_wait:
			return

		# Reset new segment flag
		if self._pending_new_segment:
			self._pending_new_segment = False

		# ── Update bigram counts (matching generate_pefa_cache_data.py logic) ──
		prev_type = None
		if prev_action:
			prev_type = self._extract_action_type(prev_action, **type_kwargs)
			# Strip _instruction suffix from prev (generator line 454: prev records
			# the structural pattern, not the instruction-specific variant)
			if prev_type and '_instruction' in prev_type:
				prev_type = prev_type.replace('_instruction', '')

		# Unconditioned bigram: full_bigrams[prev][next] += 1
		if prev_type not in self.full_bigrams:
			self.full_bigrams[prev_type] = {}
		if selected_type not in self.full_bigrams[prev_type]:
			self.full_bigrams[prev_type][selected_type] = 0
		self.full_bigrams[prev_type][selected_type] += 1

		# Conditioned bigram: conditioned_bigrams[ctx_key][prev][next] += 1
		ctx_key = self._compute_context_key(
			grabbed_objects=grabbed_objects,
			agents_on_same_surface=agents_on_same_surface,
			agents_in_room=agents_in_room,
			agents_on_land_surface=agents_on_land_surface,
			basket_contents=basket_contents,
			unsatisfied=unsatisfied,
			objects_on_same_surface=objects_on_same_surface,
		)
		if ctx_key not in self.conditioned_bigrams:
			self.conditioned_bigrams[ctx_key] = {}
		if prev_type not in self.conditioned_bigrams[ctx_key]:
			self.conditioned_bigrams[ctx_key][prev_type] = {}
		if selected_type not in self.conditioned_bigrams[ctx_key][prev_type]:
			self.conditioned_bigrams[ctx_key][prev_type][selected_type] = 0
		self.conditioned_bigrams[ctx_key][prev_type][selected_type] += 1

