def make_id(problem):
    """Convert problem description in to human-readable id."""
    key_value_strs = [f"{key}={value}" for key, value in problem.items()]

    return ",".join(key_value_strs).replace(" ", "")
