#!/bin/bash

# 모든 GPT-5 모델로 실험을 순차적으로 실행하는 통합 스크립트

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Branch: ${BRANCH}"

MODELS=("gpt-5-nano-2025-08-07" "gpt-5-mini-2025-08-07" "gpt-5-2025-08-07")
SHORT_NAMES=("gpt5-nano" "gpt5-mini" "gpt5")

# 환경별 태스크 정의
declare -A ENV_TASKS
ENV_TASKS[env0]="2 4 9 10 11 16 20"
ENV_TASKS[env1]="1 3 7 8 11 16 20"
ENV_TASKS[env2]="3 5 6 7 10 16 17"
ENV_TASKS[env3]="2 4 6 7 10 17 19"
ENV_TASKS[env4]="0 1 7 10 12 17 18 19"

ENVS=("env0" "env1" "env2" "env3" "env4")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    SHORT="${SHORT_NAMES[$i]}"
    echo "============================================"
    echo "Starting experiments with ${SHORT} (${MODEL})"
    echo "============================================"

    for ENV in "${ENVS[@]}"; do
        TASKS="${ENV_TASKS[$ENV]}"
        echo "Running ${ENV} tasks [${TASKS}] with ${SHORT}..."
        python main.py --env "$ENV" --task $TASKS --lm_id "$MODEL" --branch "$BRANCH" --source openai
    done

    echo "${SHORT} experiments completed!"
    echo ""
done

echo "All experiments completed!"
