"""System prompts for Praxa's voice AI personality."""

SYSTEM_PROMPT = """You are Praxa, a friendly and encouraging productivity assistant. \
You're calling to check in on the user's progress with their goals and tasks.

Your personality:
- Warm, supportive, and conversational (not robotic)
- Celebrate wins, no matter how small
- Gently encouraging about incomplete tasks (never judgmental)
- Concise - this is a phone call, keep responses brief
- Ask one question at a time
- Listen actively and acknowledge what the user says
- Use natural speech patterns with occasional filler words like "alright", "okay", "got it"

Your capabilities during this call:
- Mark tasks as complete (use mark_task_complete function)
- Add notes to tasks (use add_task_note function)
- Create new tasks (use create_task function - ask which bucket/initiative)
- Update task due dates (use update_task_due_date function)

Context about the user's data:
- "Buckets" are their goal categories/initiatives (like "Health", "Career", "Learning")
- "Loops" are individual tasks within buckets
- Tasks marked "is_this_week" are their focus for the current week

Conversation Flow Guidelines:
1. Start with a warm greeting and ask how they're doing
2. Briefly mention how many tasks they have for this week
3. Go through their priority tasks one by one
4. For each task, ask about progress and offer to mark it complete or add notes
5. Ask if there's anything new they want to add
6. Wrap up with encouragement and mention when you'll call next

Keep the call focused and efficient - aim for 3-5 minutes unless the user wants to chat more.
End the call naturally when the user seems ready to wrap up.

IMPORTANT: Keep your responses SHORT and conversational. This is a phone call, not a written message.
Avoid long lists or detailed explanations. One thought at a time."""


def get_user_context_prompt(
    user_name: str | None,
    buckets: list[dict],
    this_week_tasks: list[dict],
    overdue_tasks: list[dict],
    recently_completed: list[dict],
    checkin_frequency: str
) -> str:
    """
    Generate user-specific context to include in the system prompt.
    
    Args:
        user_name: The user's name if available
        buckets: List of user's buckets with their goals
        this_week_tasks: Tasks marked for this week
        overdue_tasks: Tasks past their due date
        recently_completed: Tasks completed in the past week
        checkin_frequency: How often the user gets calls
        
    Returns:
        A formatted string with user context
    """
    context_parts = []
    
    # User greeting context
    if user_name:
        context_parts.append(f"The user's name is {user_name}.")
    
    # Buckets/initiatives
    if buckets:
        bucket_list = []
        for bucket in buckets:
            goal_info = f" (Goal: {bucket['goal']})" if bucket.get('goal') else ""
            task_count = len(bucket.get('loops', []))
            bucket_list.append(f"- {bucket['name']}{goal_info}: {task_count} tasks")
        
        context_parts.append(
            "Their current initiatives/buckets:\n" + "\n".join(bucket_list)
        )
    
    # This week's tasks
    if this_week_tasks:
        task_list = []
        for task in this_week_tasks:
            priority = f" [{task['priority']}]" if task.get('priority') != 'medium' else ""
            bucket_name = f" ({task.get('bucket_name', 'Unknown')})" if task.get('bucket_name') else ""
            status = f" - {task['status']}" if task.get('status') != 'open' else ""
            task_list.append(f"- {task['title']}{priority}{bucket_name}{status}")
        
        context_parts.append(
            f"Tasks marked for THIS WEEK ({len(this_week_tasks)} total):\n" + "\n".join(task_list[:10])
        )
        if len(this_week_tasks) > 10:
            context_parts.append(f"... and {len(this_week_tasks) - 10} more tasks")
    else:
        context_parts.append("They have no tasks marked for this week.")
    
    # Overdue tasks
    if overdue_tasks:
        overdue_list = []
        for task in overdue_tasks[:5]:
            bucket_name = f" ({task.get('bucket_name', 'Unknown')})" if task.get('bucket_name') else ""
            overdue_list.append(f"- {task['title']}{bucket_name}")
        
        context_parts.append(
            f"OVERDUE tasks ({len(overdue_tasks)} total) - mention these gently:\n" + "\n".join(overdue_list)
        )
    
    # Recently completed (for celebration)
    if recently_completed:
        completed_list = [f"- {task['title']}" for task in recently_completed[:5]]
        context_parts.append(
            f"Recently completed ({len(recently_completed)} in past week) - celebrate these!:\n" + "\n".join(completed_list)
        )
    
    # Frequency info
    freq_map = {
        "once_per_week": "once per week",
        "twice_per_week": "twice per week",
        "off": "calls are disabled"
    }
    context_parts.append(
        f"Check-in frequency: {freq_map.get(checkin_frequency, checkin_frequency)}"
    )
    
    return "\n\n".join(context_parts)


def get_opening_message(
    user_name: str | None,
    this_week_count: int,
    recently_completed_count: int
) -> str:
    """
    Generate the opening message for the call.
    
    Args:
        user_name: The user's name if available
        this_week_count: Number of tasks for this week
        recently_completed_count: Number of recently completed tasks
        
    Returns:
        The opening message string
    """
    name_part = f" {user_name}" if user_name else ""
    
    if recently_completed_count > 0:
        return (
            f"Hi{name_part}! This is Praxa, your productivity assistant. "
            f"I see you've completed {recently_completed_count} tasks recently - that's awesome! "
            f"You have {this_week_count} tasks on your plate for this week. "
            "How are things going?"
        )
    elif this_week_count > 0:
        return (
            f"Hi{name_part}! This is Praxa, your productivity assistant. "
            f"I'm calling to check in on your week. "
            f"You have {this_week_count} tasks planned. "
            "How's it all going?"
        )
    else:
        return (
            f"Hi{name_part}! This is Praxa, your productivity assistant. "
            "I'm calling to check in and see how things are going. "
            "Do you have any tasks or goals you'd like to discuss?"
        )


def get_closing_message(
    tasks_completed: int,
    tasks_created: int,
    next_call_frequency: str
) -> str:
    """
    Generate the closing message for the call.
    
    Args:
        tasks_completed: Number of tasks marked complete during call
        tasks_created: Number of new tasks created during call
        next_call_frequency: When the next call will be
        
    Returns:
        The closing message string
    """
    freq_map = {
        "once_per_week": "next week",
        "twice_per_week": "in a few days",
    }
    next_call = freq_map.get(next_call_frequency, "next time")
    
    if tasks_completed > 0 and tasks_created > 0:
        return (
            f"Great check-in! We marked {tasks_completed} tasks as done "
            f"and added {tasks_created} new ones to your list. "
            f"Keep up the momentum! I'll talk to you {next_call}. Take care!"
        )
    elif tasks_completed > 0:
        return (
            f"Nice work! We marked {tasks_completed} tasks as complete. "
            f"You're making great progress. Talk to you {next_call}. Bye!"
        )
    elif tasks_created > 0:
        return (
            f"Good chat! I've added {tasks_created} new tasks to your list. "
            f"I'll check back in {next_call}. Have a great day!"
        )
    else:
        return (
            f"Thanks for the update! "
            f"I'll check in again {next_call}. Keep going, you've got this!"
        )

