#!/usr/bin/env python3
"""
Generate unified action cache data from PEFA agent JSONL logs.
Produces a single cache_data.json per agent type with:
  - Selection rates (action_type -> chosen/total/rate)
  - Unconditioned bigrams (prev_action_type -> next_action_type -> count)
  - Context-conditioned bigrams keyed by binary state features:
      - has_colocated_agent (0/1): another agent on same surface / in room / on land surface
      - basket_has_items (0/1): quadrotor basket is non-empty
      - has_grabbed (0/1): agent is currently holding an object
      - goal_object_on_same_surface (0/1): a goal-relevant object is on the same surface (robot_arm)

Usage:
  python generate_pefa_cache_data.py --results_dir ./results/baseline/gpt-5-2025-08-07
"""

import re
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_action_type(action_str, step=None):
    """
    Extract action type from action string, subdividing by target context.
    
    '[grab] <bread>(26)'                         -> 'grab_target' if bread in goal instruction
    '[putinto] <bread>(26) into <basket>(25)'    -> 'put_basket'
    '[putinto] <bread>(26) into <plate>(51)'     -> 'putinto_goal'
    '[puton] <bread>(26) on <table>(13)'         -> 'puton_goal'
    
    Args:
        action_str: Full action string
        step: Step dict from JSONL log (optional, used for grab_target detection)
    """
    if not action_str:
        return None
    m = re.match(r'\[(\w+)\]', action_str)
    if not m:
        return None
    base_type = m.group(1)

    has_instruction = bool(step and step.get('last_instruction'))

    if base_type == 'movetowards':
        target_match = re.search(r'<([^>]+)>\((\d+)\)', action_str)
        if target_match and step:
            target_name = target_match.group(1)
            target_id = int(target_match.group(2))
            target_full = target_match.group(0)
            # Check drone basket
            drone_baskets = step.get('_drone_basket_ids', set())
            if drone_baskets and target_id in drone_baskets:
                return 'movetowards_drone_basket'
            # Check agent surface
            agent_surfaces = step.get('_agent_surfaces', set())
            if target_full in agent_surfaces:
                return 'movetowards_agent_surface'
            # Check goal subject (thing to grab) or goal destination (place to put)
            goal_subjects = step.get('_goal_subjects', set())
            goal_objects = step.get('_goal_objects', set())
            if target_full in goal_subjects:
                return 'movetowards_goal_subj'
            if target_full in goal_objects:
                # Subdivide by purpose: delivering (has grabbed) vs preparing (e.g. open)
                grabbed = step.get('grabbed_object')
                if grabbed:
                    return 'movetowards_goal_obj'
                else:
                    return 'movetowards_goal_obj_open'
            # Check door
            if 'door' in target_name.lower():
                return 'movetowards_door'
            # Check if target is mentioned in oracle instruction (not goal-related)
            instr = step.get('last_instruction', '') or ''
            if target_full in instr:
                return 'movetowards_instruction'
        return 'movetowards'

    if base_type == 'grab':
        # Check if grabbed object is a goal subject (from timeline.jsonl task_goal)
        obj_match = re.search(r'<[^>]+>\(\d+\)', action_str)
        if obj_match and step:
            goal_subjects = step.get('_goal_subjects', set())
            if obj_match.group(0) in goal_subjects:
                return 'grab_goal_subj'
        return 'grab'

    if base_type == 'land_on':
        surface_match = re.search(r'<[^>]+>\(\d+\)', action_str)
        if surface_match and step:
            # Check if target surface has another agent (robot_arm)
            agent_surfaces = step.get('_agent_surfaces', set())
            if surface_match.group(0) in agent_surfaces:
                return 'land_on_agent_surface'
            # Check if target surface is mentioned in oracle instruction
            instr = step.get('last_instruction', '') or ''
            if surface_match.group(0) in instr:
                return 'land_on_instruction'
        return 'land_on'

    if base_type == 'putinto':
        # Check if target is basket (transport) vs goal (delivery)
        target_match = re.search(r'into <[^>]+>\((\d+)\)', action_str)
        if target_match and 'basket' in action_str.lower():
            basket_id = int(target_match.group(1))
            drone_baskets = step.get('_drone_basket_ids', set()) if step else set()
            if drone_baskets and basket_id in drone_baskets:
                return 'put_drone_basket'
            return 'put_basket'
        # Check if destination is a goal object
        dest_match = re.search(r'into (<[^>]+>\(\d+\))', action_str)
        if dest_match and step:
            goal_objects = step.get('_goal_objects', set())
            if dest_match.group(1) in goal_objects:
                return 'put_goal_obj'
        return 'putinto'

    if base_type == 'puton':
        # Check if destination is a goal object
        dest_match = re.search(r'on (<[^>]+>\(\d+\))', action_str)
        if dest_match and step:
            goal_objects = step.get('_goal_objects', set())
            if dest_match.group(1) in goal_objects:
                return 'put_goal_obj'
        return 'puton'

    if base_type == 'takeoff_from':
        if step:
            instr = step.get('last_instruction', '') or ''
            if 'takeoff' in instr.lower():
                return 'takeoff_from_instruction'
        return 'takeoff_from'

    return base_type


def extract_primary_target(action_str):
    """Extract the first <object>(id) from an action string for target continuity tracking.
    
    '[movetowards] <door>(7)'                        -> '<door>(7)'
    '[putinto] <bread>(26) into <basket>(25)'         -> '<bread>(26)'
    '[grab] <cup>(19)'                                -> '<cup>(19)'
    """
    if not action_str:
        return None
    m = re.search(r'<[^>]+>\(\d+\)', action_str)
    return m.group(0) if m else None


def get_agent_type_from_filename(filename):
    """'agent_robot_arm_24.jsonl' -> 'robot_arm'."""
    m = re.match(r'agent_(.+)_\d+\.jsonl', filename)
    if m:
        return m.group(1)
    return None


def compute_context_key(step, agent_type):
    """
    Compute binary context features for a step, returning a dict of 0/1 values.
    Features vary by agent type.
    """
    ctx = {}

    # --- has_colocated_agent ---
    if agent_type == 'robot_arm':
        colocated = step.get('agents_on_same_surface', [])
        ctx['has_colocated_agent'] = 1 if len(colocated) > 0 else 0
    elif agent_type == 'robot_dog':
        colocated = step.get('agents_in_room', [])
        ctx['has_colocated_agent'] = 1 if len(colocated) > 0 else 0
    elif agent_type == 'quadrotor':
        colocated = step.get('agents_on_land_surface', [])
        ctx['has_colocated_agent'] = 1 if len(colocated) > 0 else 0
    else:
        ctx['has_colocated_agent'] = 0

    # --- basket_has_items (quadrotor only) ---
    if agent_type == 'quadrotor':
        basket = step.get('basket_contents', [])
        ctx['basket_has_items'] = 1 if len(basket) > 0 else 0

    # --- has_grabbed (not for quadrotor) ---
    if agent_type != 'quadrotor':
        grabbed = step.get('grabbed_object')
        ctx['has_grabbed'] = 1 if grabbed else 0

    # --- goal_object_on_same_surface (robot_arm only) ---
    if agent_type == 'robot_arm':
        unsatisfied = step.get('unsatisfied', {})
        objects_on_surface = step.get('objects_on_same_surface', [])
        # Extract goal object names from unsatisfied keys
        goal_objects = set()
        for key in unsatisfied.keys():
            # e.g. inside_<bread>(26)_<plate>(51) -> <bread>(26), <plate>(51)
            for obj_match in re.finditer(r'<[^>]+>\(\d+\)', key):
                goal_objects.add(obj_match.group(0))
        # Check if any goal object is on same surface
        surface_set = set(objects_on_surface) if objects_on_surface else set()
        has_goal_on_surface = 1 if goal_objects & surface_set else 0
        ctx['goal_object_on_same_surface'] = has_goal_on_surface

    return ctx


def context_to_key(ctx):
    """Convert context dict to a hashable string key like 'c1_b0_g1'."""
    parts = []
    for k in sorted(ctx.keys()):
        short = k[0]  # first letter: h, b, g
        # Use more readable short names
        if k == 'has_colocated_agent':
            short = 'c'
        elif k == 'basket_has_items':
            short = 'b'
        elif k == 'has_grabbed':
            short = 'g'
        elif k == 'goal_object_on_same_surface':
            short = 's'
        parts.append(f"{short}{ctx[k]}")
    return '_'.join(parts)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_agent_steps(results_dir, episodes=None):
    """
    Load agent JSONL files from results directory.
    
    Also collects per-episode metadata and injects into step dicts:
      - '_agent_surfaces': surfaces where robot_arm sits (for quadrotor land_on subdivision)
      - '_goal_subjects': goal subject objects from timeline.jsonl (for grab_target)
      - '_goal_objects': goal destination objects from timeline.jsonl
    
    Args:
        results_dir: Path to results directory containing env*/episode_*/agent_*.jsonl
        episodes: List of 'envX/episode_Y' strings to include.
                  If None, includes all episodes from all envs.
    
    Returns dict: {agent_type: [list of (episode_steps, episode_id)]}
    where episode_steps is list of step dicts for that agent in one episode.
    """
    results_path = Path(results_dir)
    agent_data = defaultdict(list)  # agent_type -> [(steps, episode_id), ...]
    # Collect robot_arm surfaces per episode for land_on subdivision
    episode_agent_surfaces = defaultdict(set)  # episode_id -> {surface_str, ...}
    # Collect task_goal per episode from timeline.jsonl
    episode_goal_subjects = defaultdict(set)   # episode_id -> {subject_str, ...}
    episode_goal_objects = defaultdict(set)     # episode_id -> {object_str, ...}
    # Collect drone basket IDs per episode for put_drone_basket subdivision
    episode_drone_basket_ids = defaultdict(set)  # episode_id -> {basket_id_int, ...}

    # Build set of allowed episode paths for fast lookup
    allowed = None
    if episodes is not None:
        allowed = set(episodes)

    for env_dir in sorted(results_path.glob('env*')):
        for episode_dir in sorted(
            env_dir.glob('episode_*'),
            key=lambda p: int(p.name.split('_')[1])
        ):
            episode_id = f"{env_dir.name}/{episode_dir.name}"
            if allowed is not None and episode_id not in allowed:
                continue

            # Read task_goal from timeline.jsonl
            timeline_file = episode_dir / 'timeline.jsonl'
            if timeline_file.exists():
                with open(timeline_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        task_goal = json.loads(first_line).get('task_goal', {})
                        for key in task_goal.keys():
                            km = re.match(r'(.+?)_(<[^>]+>\(\d+\))_(<[^>]+>\(\d+\))', key)
                            if km:
                                episode_goal_subjects[episode_id].add(km.group(2))
                                episode_goal_objects[episode_id].add(km.group(3))

            for agent_file in sorted(episode_dir.glob('agent_*.jsonl')):
                agent_type = get_agent_type_from_filename(agent_file.name)
                if agent_type is None:
                    continue

                steps = []
                with open(agent_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            steps.append(json.loads(line))

                if steps:
                    agent_data[agent_type].append((steps, episode_id))

                # Collect surfaces where robot_arm sits
                if agent_type == 'robot_arm':
                    for s in steps:
                        surface = s.get('on_surface')
                        if surface:
                            episode_agent_surfaces[episode_id].add(surface)

                # Collect drone basket IDs from quadrotor logs
                if agent_type == 'quadrotor':
                    for s in steps:
                        bid = s.get('basket_id')
                        if bid is not None:
                            episode_drone_basket_ids[episode_id].add(bid)
                            break  # one basket per quadrotor

    # Inject per-episode metadata into all steps
    for agent_type, episodes_list in agent_data.items():
        for steps, episode_id in episodes_list:
            goal_subs = episode_goal_subjects.get(episode_id, set())
            goal_objs = episode_goal_objects.get(episode_id, set())
            surfaces = episode_agent_surfaces.get(episode_id, set())
            drone_baskets = episode_drone_basket_ids.get(episode_id, set())
            for s in steps:
                s['_goal_subjects'] = goal_subs
                s['_goal_objects'] = goal_objs
                s['_agent_surfaces'] = surfaces
                s['_drone_basket_ids'] = drone_baskets

    return dict(agent_data)


# ── Analysis functions ───────────────────────────────────────────────────────

def compute_selection_rates(all_steps, agent_type):
    """
    Compute action selection rates.
    
    If available_actions is present in step metadata:
      Denominator: # steps where that action type appeared in available_actions
      Numerator: # times that action type was actually selected
    
    Fallback (no available_actions):
      Denominator: total_steps
      Numerator: # times that action type was selected
    """
    # Check if available_actions exists in logs
    has_available = False
    for episode_steps, _ in all_steps:
        for step in episode_steps:
            if step.get('available_actions'):
                has_available = True
                break
        if has_available:
            break

    if has_available:
        # Accurate calculation using available_actions
        chosen_counts = Counter()   # action_type -> times selected
        avail_counts = Counter()    # action_type -> times appeared in available_actions

        for episode_steps, _ in all_steps:
            for step in episode_steps:
                selected_action = step.get('action', '')
                selected_type = extract_action_type(selected_action, step=step)

                # Count available action types at this step
                available = step.get('available_actions', [])
                seen_types = set()
                for avail_action in available:
                    atype = extract_action_type(avail_action, step=step)
                    if atype and atype != 'wait':
                        seen_types.add(atype)
                for atype in seen_types:
                    avail_counts[atype] += 1

                # Count selected
                if selected_type and selected_type != 'wait':
                    chosen_counts[selected_type] += 1

        stats = {}
        for atype in avail_counts:
            chosen = chosen_counts.get(atype, 0)
            total = avail_counts[atype]
            stats[atype] = {
                "chosen": chosen,
                "total": total,
                "rate": round(chosen / total, 6) if total > 0 else 0.0
            }
        return stats
    else:
        # Fallback: total_steps as denominator
        total_steps = sum(len(steps) for steps, _ in all_steps)
        action_type_counts = Counter()
        for episode_steps, _ in all_steps:
            for step in episode_steps:
                action = step.get('action', '')
                atype = extract_action_type(action, step=step)
                if atype and atype != 'wait':
                    action_type_counts[atype] += 1

        stats = {}
        for atype, count in action_type_counts.items():
            stats[atype] = {
                "chosen": count,
                "total": total_steps,
                "rate": round(count / total_steps, 6) if total_steps > 0 else 0.0
            }
        return stats


def compute_bigrams(all_steps, agent_type, conditioned=False):
    """
    Compute bigram frequencies: prev_action_type -> next_action_type.
    
    If conditioned=True, returns dict of context_key -> Counter of (prev, next) bigrams.
    If conditioned=False, returns single Counter of (prev, next) bigrams.
    """
    if conditioned:
        conditioned_bigrams = defaultdict(Counter)
    else:
        bigrams = Counter()

    def _record(prev_t, next_t, step_for_ctx):
        if conditioned:
            ctx = compute_context_key(step_for_ctx, agent_type)
            ctx_key = context_to_key(ctx)
            conditioned_bigrams[ctx_key][(prev_t, next_t)] += 1
        else:
            bigrams[(prev_t, next_t)] += 1

    for episode_steps, _ in all_steps:
        prev_type = 'None'
        prev_step_num = None
        prev_step = None
        for i, step in enumerate(episode_steps):
            action = step.get('action', '')
            action_type = extract_action_type(action, step=step)
            step_num = step.get('step', i)

            # Treat [wait] actions: skip wait->wait, but record prev->wait
            if action_type is None or action_type == 'wait':
                # Don't update prev_type so wait->wait is skipped
                continue

            # Check for step gap (implicit waits between recorded steps)
            has_gap = (prev_step_num is not None and step_num - prev_step_num > 1)

            if has_gap and prev_type != 'None':
                # prev_real_action -> wait (use prev step's context)
                _record(prev_type, 'wait', prev_step)
                # wait -> current_action
                _record('wait', action_type, step)
            else:
                _record(prev_type, action_type, step)

            prev_type = action_type.replace('_instruction', '') if '_instruction' in action_type else action_type
            prev_step_num = step_num
            prev_step = step

        # Episode end: last real action -> wait (if last action was not wait)
        if prev_type and prev_type != 'None' and prev_step is not None:
            _record(prev_type, 'wait', prev_step)

    if conditioned:
        return dict(conditioned_bigrams)
    return bigrams


def compute_instruction_match_rates(all_steps, agent_type):
    """
    For each executed action type, compute the proportion of times
    the plan (action verb + target) was mentioned in the oracle instruction.
    
    Matching logic:
      Check if the action's verb (plan) keyword appears in the oracle instruction.
    
    Returns:
        dict: {action_type: {matched, total, rate}}
              rate = matched / total
    """
    total_counts = Counter()
    matched_counts = Counter()

    # Map base verbs to keyword sets that may appear in oracle instructions
    VERB_KEYWORDS = {
        'movetowards': ['movetowards'],
        'grab': ['grab', 'pick', 'takeout'],
        'open': ['open'],
        'close': ['close'],
        'putinto': ['putinto', 'puton', 'place'],
        'puton': ['putinto', 'puton', 'place'],
        'takeoff_from': ['takeoff'],
        'land_on': ['land'],
    }

    for episode_steps, _ in all_steps:
        for step in episode_steps:
            action = step.get('action', '')
            atype = extract_action_type(action, step=step)
            if not atype or atype == 'wait':
                continue

            total_counts[atype] += 1
            instr = step.get('last_instruction', '') or ''
            if not instr:
                continue

            # Extract base verb from action
            verb_match = re.match(r'\[(\w+)\]', action)
            if not verb_match:
                continue
            verb = verb_match.group(1)

            keywords = VERB_KEYWORDS.get(verb, [verb])
            instr_lower = instr.lower()
            if any(kw in instr_lower for kw in keywords):
                matched_counts[atype] += 1

    stats = {}
    for atype in total_counts:
        matched = matched_counts.get(atype, 0)
        total = total_counts[atype]
        stats[atype] = {
            "matched": matched,
            "total": total,
            "rate": round(matched / total, 6) if total > 0 else 0.0
        }
    return stats


def bigrams_to_grouped(bigrams):
    """Convert Counter of (prev, next) -> count to grouped dict {prev: [[next, count], ...]}."""
    grouped = defaultdict(list)
    for (prev, next_), count in bigrams.items():
        grouped[prev].append([next_, count])
    for prev in grouped:
        grouped[prev].sort(key=lambda x: x[1], reverse=True)
    return dict(grouped)


# ── Output ───────────────────────────────────────────────────────────────────

def build_cache_json(agent_type, all_steps):
    """Build the unified cache data structure for one agent type."""
    # Selection rates
    rates = compute_selection_rates(all_steps, agent_type)

    # Both conditioned and unconditioned bigrams
    cond_bigrams = compute_bigrams(all_steps, agent_type, conditioned=True)
    full_bigrams = compute_bigrams(all_steps, agent_type, conditioned=False)

    # Instruction match rates
    instr_match = compute_instruction_match_rates(all_steps, agent_type)

    data = {
        "agent_type": agent_type,
        "total_episodes": len(all_steps),
        "total_steps": sum(len(steps) for steps, _ in all_steps),
        "selection_rates": rates,
        "instruction_match_rates": instr_match,
        "bigrams": bigrams_to_grouped(full_bigrams),
        "conditioned_bigrams": {},
        "context_features": [],
    }

    # Document which context features are used for this agent type
    sample_step = all_steps[0][0][0] if all_steps and all_steps[0][0] else {}
    sample_ctx = compute_context_key(sample_step, agent_type)
    data["context_features"] = sorted(sample_ctx.keys())

    # Convert conditioned bigrams
    for ctx_key, bigram_counter in cond_bigrams.items():
        data["conditioned_bigrams"][ctx_key] = bigrams_to_grouped(bigram_counter)

    return data


def write_report(data, output_path):
    """Write human-readable report alongside JSON."""
    with open(output_path, 'w') as f:
        f.write(f"Agent Type: {data['agent_type']}\n")
        f.write(f"Episodes: {data['total_episodes']}, Steps: {data['total_steps']}\n")
        f.write("=" * 80 + "\n\n")

        # Selection rates
        f.write("SELECTION RATES\n")
        f.write("-" * 40 + "\n")
        sorted_rates = sorted(data['selection_rates'].items(),
                              key=lambda x: x[1]['rate'], reverse=True)
        for atype, s in sorted_rates:
            f.write(f"  {atype}: {s['chosen']}/{s['total']} ({s['rate']*100:.1f}%)\n")
        f.write("\n")

        # Instruction match rates
        if 'instruction_match_rates' in data and data['instruction_match_rates']:
            f.write("INSTRUCTION MATCH RATES (plan in oracle instruction)\n")
            f.write("-" * 40 + "\n")
            sorted_match = sorted(data['instruction_match_rates'].items(),
                                  key=lambda x: x[1]['rate'], reverse=True)
            for atype, s in sorted_match:
                f.write(f"  {atype}: {s['matched']}/{s['total']} ({s['rate']*100:.1f}%)\n")
            f.write("\n")

        # Full bigrams
        f.write("FULL BIGRAMS\n")
        f.write("-" * 40 + "\n")
        full_grouped = data.get('bigrams', {})
        full_total = sum(c for nexts in full_grouped.values() for _, c in nexts)
        f.write(f"  ({full_total} total)\n")
        for prev_type, nexts in sorted(full_grouped.items()):
            total = sum(c for _, c in nexts)
            for next_type, count in nexts:
                pct = 100.0 * count / total if total > 0 else 0
                f.write(f"    [{prev_type} -> {next_type}]: {count} ({pct:.1f}%)\n")
        f.write("\n")

        # Conditioned bigrams
        f.write("CONDITIONED BIGRAMS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Context features: {data['context_features']}\n\n")
        for ctx_key in sorted(data['conditioned_bigrams'].keys()):
            grouped = data['conditioned_bigrams'][ctx_key]
            total_in_ctx = sum(c for nexts in grouped.values() for _, c in nexts)
            f.write(f"  [{ctx_key}] ({total_in_ctx} total)\n")
            for prev_type, nexts in sorted(grouped.items()):
                total = sum(c for _, c in nexts)
                for next_type, count in nexts:
                    pct = 100.0 * count / total if total > 0 else 0
                    f.write(f"    [{prev_type} -> {next_type}]: {count} ({pct:.1f}%)\n")
            f.write("\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Default: earliest episode per env0-env3 where all 3 agent types have data
    # (similar line counts 12-14, unbiased selection)
    DEFAULT_EPISODES = [
        'env0/episode_15',   # quad=6, arm=2, dog=6, total=14
        'env1/episode_10',   # quad=6, arm=2, dog=6, total=14
        'env2/episode_11',   # quad=6, arm=2, dog=4, total=12
        'env3/episode_16',   # quad=6, arm=2, dog=4, total=12
    ]

    parser = argparse.ArgumentParser(description='Generate PEFA cache data from agent logs')
    parser.add_argument('--results_dir', type=str,
                        default='./results/cache/gpt-5-2025-08-07',
                        help='Path to results directory containing env*/episode_*/agent_*.jsonl')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for cache files (default: current dir)')
    parser.add_argument('--episodes', type=str, nargs='*',
                        default=DEFAULT_EPISODES,
                        help='Which episodes to include as envX/episode_Y')
    parser.add_argument('--all', action='store_true',
                        help='Use ALL episodes from all envs (ignores --episodes)')
    args = parser.parse_args()

    agent_data = load_agent_steps(
        args.results_dir,
        episodes=None if args.all else args.episodes,
    )

    print(f"Loaded data for agent types: {list(agent_data.keys())}")
    for agent_type, episodes in agent_data.items():
        total_steps = sum(len(steps) for steps, _ in episodes)
        print(f"  {agent_type}: {len(episodes)} episodes, {total_steps} steps")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for agent_type, episodes in agent_data.items():
        data = build_cache_json(agent_type, episodes)

        json_path = output_dir / f"cache_data_{agent_type}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Written: {json_path}")

        report_path = output_dir / f"cache_report_{agent_type}.txt"
        write_report(data, report_path)
        print(f"  Written: {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
