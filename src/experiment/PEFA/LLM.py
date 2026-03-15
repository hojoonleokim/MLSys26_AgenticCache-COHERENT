
import copy
import openai
import re
import json
from openai import OpenAIError, OpenAI
import backoff
import os

def remove_all_brackets(text):
	"""Remove bracket symbols but keep their content: [], <>, ()"""
	text = re.sub(r'\[([^\]]*)\]', lambda m: m.group(1).strip(), text)  # [ content ] → content
	text = re.sub(r'<([^>]*)>', lambda m: m.group(1).strip(), text)     # < content > → content
	text = re.sub(r'\(([^)]*)\)', lambda m: m.group(1).strip(), text)   # ( content ) → content
	return text

class LLM:
	def __init__(self, source, lm_id, args, agent_node):

		self.args = args
		self.debug = args.debug
		self.source = args.source
		self.lm_id = lm_id
		self.chat = True
		self.total_cost = 0
		self.device = None
		# Logging: always use args.lm_id (target model) for directory
		log_base = f'./results/{args.branch}/{args.lm_id}/{args.env}'
		os.makedirs(log_base, exist_ok=True)
		agent_label = f'{agent_node["class_name"]}_{agent_node["id"]}'
		self.record_dir = os.path.join(log_base, f'{args.env}_{agent_label}_{self.lm_id}_log.txt')
		self.LLM_token_record_dir = os.path.join(log_base, f'{agent_label}_{self.lm_id}_token.txt')

		if self.source == 'openai':

			api_key = args.api_key  # your openai api key
			organization= args.organization # your openai organization

			client = OpenAI(api_key = api_key, organization=organization)
			if self.chat:
				self.sampling_params = {
					# "max_tokens": args.max_tokens,
                    # "temperature": args.t,
                    # "top_p": 1.0,
                    "n": args.n
				}

		def lm_engine(source, lm_id, device):
			
			@backoff.on_exception(backoff.expo, OpenAIError)
			def _generate(prompt, sampling_params):
				usage = 0
				if source == 'openai':
					try:
						if self.chat:
							prompt.insert(0,{"role":"system", "content":"You are a helper assistant."})
							response = client.chat.completions.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
							if self.debug:
								with open(f"./chat_raw.json", 'a') as f:
									f.write(json.dumps(response, indent=4))
									f.write('\n')
							generated_samples = [response.choices[i].message.content for i in
                                                    range(sampling_params['n'])]
							
							# Extract and log token usage
							prompt_tokens = response.usage.prompt_tokens
							completion_tokens = response.usage.completion_tokens
							total_tokens = response.usage.total_tokens
							self.write_token_log_to_file(f"INPUT: {prompt_tokens}, OUTPUT: {completion_tokens}, TOTAL: {total_tokens}")
							
							usage = response.usage.total_tokens * 0.01 / 1000
							if 'gpt-4-0125-preview' in self.lm_id:
								usage = response.usage.prompt_tokens * 0.01 / 1000 + response.usage.completion_tokens * 0.03 / 1000
							elif 'gpt-3.5-turbo-1106' in self.lm_id:
								usage = response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
						# mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
						# 				  range(sampling_params['n'])]
						else:
							raise ValueError(f"{lm_id} not available!")
					except OpenAIError as e:
						print(e)
						raise e
				
				else:
					raise ValueError("invalid source")
				return generated_samples, usage

			return _generate

		self.generator = lm_engine(self.source, self.lm_id, self.device)

	def parse_answer(self, available_actions, text):
		
		# "YES I CAN." 제거
		text = re.sub(r'YES\sI\sCAN\.?', '', text)
		
		text = text.replace("_", " ")
		text = text.replace("takeoff from", "takeoff_from")
		text = text.replace("land on", "land_on")

		for i in range(len(available_actions)):
			action = available_actions[i]
			if action in text:
				return action
		self.write_log_to_file('\nThe first action parsing failed!!!')

		for i in range(len(available_actions)):
			action = available_actions[i]
			option = chr(ord('A') + i)
			if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text.split(' ') or f"{option})" in text.split(' '):
				return action
		self.write_log_to_file('\nThe second action parsing failed!!!')

		removed_text = remove_all_brackets(text)
		for i in range(len(available_actions)):
			action = remove_all_brackets(available_actions[i])
			if action in removed_text:
				return available_actions[i]
		self.write_log_to_file('\nThe third action parsing failed!!!')

		for i in range(len(available_actions)):
			action = available_actions[i]
			option = chr(ord('A') + i)
			words = removed_text.split()  # 공백으로 분해
			if len(words) == 1 and option in words:
				return action
		self.write_log_to_file('\nThe fourth action parsing failed!!!')

		print("WARNING! No available action parsed!!! Output plan NONE!\n")
		return None

	def get_available_plans(self, agent_node, next_rooms, all_landable_surfaces, landable_surfaces, on_surfaces, 
						 grabbed_objects, reached_objects, unreached_objecs, on_same_surface_objects
						 ):
		"""
		'quadrotor':
		[land_on] <surface>
		[movetowards] <surface>/<next_room>
		[takeoff_from] <surface>

		'robot dog':
		[open] <container>/<door>
		[close] <container>/<door>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>
		[movetowards] <object>

		'robot arm':
		[open] <container>
		[close] <container>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>

		"""
		available_plans = []
		if agent_node["class_name"] == "quadrotor":
			other_landable_surfaces = []
			if "FLYING" in agent_node["states"]:
				if landable_surfaces is not None:
					available_plans.append(f"[land_on] <{landable_surfaces['class_name']}>({landable_surfaces['id']})")
					# Remove landable_surfaces by comparing id instead of object identity
					other_landable_surfaces = [s for s in all_landable_surfaces if s['id'] != landable_surfaces['id']]
				else:
					other_landable_surfaces = copy.deepcopy(all_landable_surfaces)
				if len(other_landable_surfaces) != 0:
					for surface in other_landable_surfaces :
						available_plans.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
				for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

			if "LAND" in agent_node["states"]:
				if on_surfaces is not None:
					available_plans.append(f"[takeoff_from] <{on_surfaces['class_name']}>({on_surfaces['id']})")

		if agent_node["class_name"] == "robot dog" or agent_node["class_name"] == "robot_dog":
			# if grabbed_objects is not None:
			# 	available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			# The robotic dog is not allowed to put things on the floor. If it needs to open the door and has something in its hand, it needs to find a low surface to put things first
			if len(reached_objects) != 0:
				for reached_object in reached_objects:
					if grabbed_objects is None:
						if 'CONTAINERS' in reached_object['properties'] and 'CLOSED' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'CLOSED' in reached_object['states']:
							available_plans.append(f"[open] <{reached_object['class_name']}>({reached_object['id']})")
						if 'CONTAINERS' in reached_object['properties'] and 'OPEN' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'OPEN' in reached_object['states']:
							available_plans.append(f"[close] <{reached_object['class_name']}>({reached_object['id']})")
						if 'GRABABLE' in reached_object['properties']:
							available_plans.append(f"[grab] <{reached_object['class_name']}>({reached_object['id']})")
					if grabbed_objects is not None:
						if 'CONTAINERS' in reached_object['properties'] and ('OPEN' in reached_object['states'] or "OPEN_FOREVER" in reached_object['states']):
							available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{reached_object['class_name']}>({reached_object['id']})")
						if 'SURFACES' in reached_object['properties']:
							available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{reached_object['class_name']}>({reached_object['id']})")
			
			if len(unreached_objecs) != 0:
				for unreached_object in unreached_objecs:
					available_plans.append(f"[movetowards] <{unreached_object['class_name']}>({unreached_object['id']})")
			for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")


		if agent_node['class_name'] == 'robot arm' or agent_node['class_name'] == 'robot_arm':
			if grabbed_objects is not None:
				available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			for on_same_surface_object in on_same_surface_objects:
				if grabbed_objects is None:
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'OPEN' in on_same_surface_object['states']:
						available_plans.append(f"[close] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'CLOSED' in on_same_surface_object['states']:
						available_plans.append(f"[open] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'GRABABLE' in on_same_surface_object['properties']:
						available_plans.append(f"[grab] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

				if grabbed_objects is not None:
					
					if 'CONTAINERS' in on_same_surface_object['properties'] and ('OPEN' in on_same_surface_object['states'] or "OPEN_FOREVER" in on_same_surface_object['states']):
						available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'SURFACES' in on_same_surface_object['properties']:
						available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

		# Add wait action at the end for all agent types
		available_plans.append("[wait]")

		plans = ""
		for i, plan in enumerate(available_plans):
			plans += f"{chr(ord('A') + i)}. {plan}\n"
		# print(agent_node["class_name"],agent_node['id'])
		# print(available_plans)
		return plans, len(available_plans), available_plans

			
	def run(self, agent_node, chat_agent_info, current_room, next_rooms, all_landable_surfaces,
			landable_surfaces, on_surfaces, grabbed_objects, reached_objects, unreached_objecs,
			on_same_surface_objects, action_history):

		info = {
			"num_available_actions": None,
			"prompts": None,
			"outputs": None,
			"plan": None,
			"action_list": None,
			"cost": self.total_cost,
			f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost
		}

		# 1) 프롬프트 로드 (내용은 나중에 네가 수정 가능)
		prompt_path = chat_agent_info['prompt_path']
		with open(prompt_path, 'r') as f:
			base_prompt = f.read()

		# 2) 액션 리스트 생성
		available_plans, num, available_plans_list = self.get_available_plans(
			agent_node, next_rooms, all_landable_surfaces, landable_surfaces, on_surfaces,
			grabbed_objects, reached_objects, unreached_objecs, on_same_surface_objects
		)

		# 3) 히스토리 문자열화 (최근 10개만, action_history는 dict)
		if action_history:
			sorted_history = sorted(action_history.items())[-10:]
			action_history_str = '\n'.join([f"{v} at step {k}" for k, v in sorted_history])
		else:
			action_history_str = 'None'

		# 5) 플레이스홀더 치환
		agent_prompt = base_prompt
		agent_prompt = agent_prompt.replace('#OBSERVATION#', chat_agent_info['observation'])
		agent_prompt = agent_prompt.replace('#ACTIONLIST#', available_plans)
		agent_prompt = agent_prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
		agent_prompt = agent_prompt.replace('#PLANHISTORY#', action_history_str)
		
		# 6) 최종 프롬프트 구성 (단일 쿼리)
		final_prompt = agent_prompt

		if self.debug:
			print(f"[single_query_prompt]:\n{final_prompt}")
		self.write_log_to_file("prompt_single\n" + final_prompt + "\n-----")

		# 7) 단일 호출
		chat_prompt = [{"role": "user", "content": final_prompt}]
		outputs, usage = self.generator(chat_prompt, self.sampling_params)
		output = outputs[0] if outputs else ""
		info['cot_outputs'] = outputs
		info['cost'] = self.total_cost

		if self.debug:
			print(f"[single_query_output]:\n{output}")
			print(f"[total cost]: {self.total_cost}")
		self.write_log_to_file("output_single\n" + output + "\n-----")

		# 8) 파싱 유틸
		def _normalize(s):
			return ' '.join(s.strip().split()).lower()

		def parse_single_output(raw_text, action_list):
			txt = raw_text.strip()

			# 불가능 패턴
			lower = txt.lower()
			if lower.startswith("sorry, i cannot:") or lower.startswith("sorry, i cant:") or lower.startswith("sorry, i cannot"):
				# "SORRY, I CANNOT: 이유" 그대로 메시지로 반환
				reason = txt.split(":", 1)[1].strip() if ":" in txt else txt
				return None, f"SORRY, I CANNOT: {reason}"

			# 가능한 패턴: ACTION: ...
			if lower.startswith("action:"):
				chosen = txt.split(":", 1)[1].strip()
				# 1) exact match
				for a in action_list:
					if _normalize(a) == _normalize(chosen):
						# Convert [wait] to None, keep message
						if a.strip() == '[wait]':
							return None, chosen
						return a, None
				# 2) substring 혹은 포함 관계로 완화 매칭
				for a in action_list:
					na = _normalize(a)
					nc = _normalize(chosen)
					if na in nc or nc in na:
						# Convert [wait] to None, keep message
						if a.strip() == '[wait]':
							return None, chosen
						return a, None

			# 형식을 어긴 경우: 마지막 시도 - 액션 리스트 중 하나라도 들어있으면 채택
			for a in action_list:
				if _normalize(a) in _normalize(txt):
					# Convert [wait] to None, keep message
					if a.strip() == '[wait]':
						return None, txt
					return a, None

			# 완전히 실패하면 불가능 처리
			return None, "SORRY, I CANNOT: Could not parse a valid action from the given output."

		# 9) 결과 파싱
		plan, sorry_msg = parse_single_output(output, available_plans_list)

		if plan is not None:
			plan_str = plan
			message = f"ACTION: {plan_str}"
			info.update({
				"num_available_actions": num,
				"prompts": chat_prompt,
				"plan": plan,
				"action_list": available_plans_list,
				f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost,
				"outputs": message
			})
		else:
			# Check if this is a [wait] action (plan=None but message contains [wait])
			if sorry_msg and '[wait]' in sorry_msg.lower():
				message = f"ACTION: [wait]"
			else:
				# 불가능 메시지
				message = sorry_msg if sorry_msg else "SORRY, I CANNOT: unspecified reason."
			info.update({
				"num_available_actions": num,
				"prompts": chat_prompt,
				"plan": None,
				"action_list": available_plans_list,
				f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost,
				"outputs": message
			})

		return message, info

	def write_log_to_file(self,log_message, file_name=None):
		file_name = self.record_dir
		with open(file_name, 'a') as file:  
			file.write(log_message + '\n')  
			
	def write_token_log_to_file(self,log_message, file_name=None):
		file_name = self.LLM_token_record_dir
		with open(file_name, 'a') as file:  
			file.write(log_message + '\n')  