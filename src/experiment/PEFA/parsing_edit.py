import re

def remove_all_brackets(text):
    """Remove bracket symbols but keep their content: [], <>, ()"""
    text = re.sub(r'\[([^\]]*)\]', lambda m: m.group(1).strip(), text)  # [ content ] → content
    text = re.sub(r'<([^>]*)>', lambda m: m.group(1).strip(), text)     # < content > → content
    text = re.sub(r'\(([^)]*)\)', lambda m: m.group(1).strip(), text)   # ( content ) → content
    return text

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
            return action
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
                all_landable_surfaces.remove(landable_surfaces)
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

    plans = ""
    for i, plan in enumerate(available_plans):
        plans += f"{chr(ord('A') + i)}. {plan}\n"
    print(agent_node["class_name"],agent_node['id'])
    print(available_plans)
    return plans, len(available_plans), available_plans


def test_parsing():
    """Test function to demonstrate the improved parsing capabilities"""
    
    # Mock class for testing
    class TestParser:
        def write_log_to_file(self, message):
            print(f"LOG: {message}")
    
    parser = TestParser()
    
    # Add the parse_answer method to the test parser
    TestParser.parse_answer = parse_answer
    
    # Example complex input
    complex_text = "takeoff_from <lower livingroom floor>(1), movetowards <door>(7),B. movetowards <dining table>(30), YES I CAN. J,A. putinto wallet(34) into basket(29), A, B) movetowards <coffee table>(5), [grab] napkin(23) , H movetowards <basket>(22)"
    
    # Example available actions
    available_actions = [
        "[takeoff_from] <lower livingroom floor>(1)",
        "[movetowards] <door>(7)", 
        "[movetowards] <dining table>(30)",
        "[putinto] wallet(34) into basket(29)",
        "[movetowards] <coffee table>(5)",
        "[grab] napkin(23)",
        "[movetowards] <basket>(22)"
    ]
    
    print("Testing complex input parsing:")
    print(f"Input: {complex_text}")
    print(f"Available actions: {available_actions}")
    print()
    
    # Test parsing
    result = parser.parse_answer(available_actions, complex_text)
    print(f"Parsed result: {result}")
    
    # Test individual components
    test_cases = [
        "takeoff_from <lower livingroom floor>(1)",
        "B. movetowards <dining table>(30)",
        "A. putinto wallet(34) into basket(29)",
        "[grab] napkin(23)",
        "movetowards <basket>(22)",
        "A",  # 단순히 A만
        "B",  # 단순히 B만
        "C",  # 단순히 C만
        "A B C D E F",  # 여러 옵션이 나열된 경우
        "(A)",  # 괄호 안의 A
        "option A"  # option A 형태
    ]
    
    print("\nTesting individual components:")
    for test_case in test_cases:
        result = parser.parse_answer(available_actions, test_case)
        print(f"'{test_case}' -> {result}")

# Uncomment the line below to run the test
# test_parsing()
