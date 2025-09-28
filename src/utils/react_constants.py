TAG_QUESTION = "react_question"
TAG_THOUGHT = "react_thought"
TAG_ACTION = "react_action"
TAG_ACTION_INPUT = "react_action_input"
TAG_OBSERVATION = "react_observation"
TAG_FINAL_ANSWER = "react_final_answer"

import pendulum
# timezone mặc định (VN)
DEFAULT_TZ = pendulum.timezone("Asia/Ho_Chi_Minh")

__all__ = [
    "TAG_QUESTION",
    "TAG_THOUGHT",
    "TAG_ACTION",
    "TAG_ACTION_INPUT",
    "TAG_OBSERVATION",
    "TAG_FINAL_ANSWER",
    "DEFAULT_TZ"
]