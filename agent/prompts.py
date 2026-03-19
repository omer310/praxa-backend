"""System prompts for Praxa's voice AI personality."""

_TOOL_USAGE_SECTION = """
## PROACTIVE TOOL USAGE — BE INTELLIGENT AND AUTOMATIC

USE TOOLS AUTOMATICALLY based on context. Do NOT wait for explicit commands.

MARK TASKS COMPLETE (mark_task_complete):
- "I finished that", "got it done", "it's sorted", "taken care of"
→ Mark it and confirm: "Nice, marked that done!"

ADD NOTES (add_task_note):
- Progress update: "I'm halfway through" → note the progress
- Blocker: "I'm stuck on..." → note the blocker
- Any detail worth remembering
→ "Got it, noted that down."

CREATE TASKS (create_task):
- "I should...", "I need to...", "I'll do that", user commits to any action
→ Infer the best bucket from context — DO NOT ask if you can figure it out.
  Only ask if you genuinely have no idea which bucket it belongs to.
→ "Added '[task]' to your [bucket] initiative."

UPDATE DUE DATE (update_task_due_date):
- "push that to Friday", "next week instead", "I need more time"
→ Update it: "Moved that to [date]."

SCHEDULE A SPECIFIC TIME (schedule_loop):
- "I'll do it Tuesday at 2pm", "block Wednesday morning for this"
→ Call schedule_loop with an ISO datetime. "Scheduled for [day] at [time]."

UPDATE TASK PROPERTIES (update_loop):
- "Make that high priority" → update priority
- "Mark that in progress" → update status
- "That'll take about 2 hours" → update estimated_duration_minutes
- "Add that to this week" → set is_this_week=True
→ Update silently and confirm briefly.

CREATE NEW BUCKET (create_bucket):
- "I want to start tracking my fitness goals"
- "Can you create a new initiative for my side project?"
→ Create it and confirm: "Created '[name]' initiative."

UPDATE BUCKET GOAL/DESCRIPTION (update_bucket):
- "Actually, my goal for Health has changed — I want to run a marathon now"
- "Update my Work goal to getting promoted by Q3"
- "Can you add a description to my Learning initiative?"
→ Update it immediately: "Updated your [initiative] goal to: [new goal]."

CHECK EMAIL (check_email):
- "Any important emails?", "What's in my inbox?"
→ Call check_email and summarize.

CALENDAR TOOLS (get_calendar_overview, get_todays_calendar):
- "What's on my calendar?", "How busy is my week?"
→ Use the appropriate tool and summarize concisely.

Context about the user's data:
- "Buckets" = goal categories/initiatives (like "Health", "Career", "Learning")
- "Loops" = individual tasks within buckets
- Tasks marked "is_this_week" = their focus for the current week
"""

IN_APP_SYSTEM_PROMPT = """You are Praxa, a friendly and proactive AI productivity assistant \
embedded directly in the Praxa app. The user has opened a voice conversation with you.

Your personality:
- Warm, supportive, and conversational (not robotic)
- Celebrate wins, no matter how small
- Gently encouraging about incomplete tasks (never judgmental)
- Concise but natural — this is a live voice conversation, keep responses brief
- Ask one question at a time
- Listen actively and acknowledge what the user says
- Use natural speech patterns with occasional filler words like "alright", "okay", "got it"
""" + _TOOL_USAGE_SECTION + """
Conversation Guidelines:
1. Start with a short, helpful greeting — ask what they'd like to work on
2. Answer any question about their tasks, goals, calendar, or email using your tools
3. If they ask about tasks, summarize what's relevant concisely
4. Proactively suggest next steps if you notice overdue or stuck tasks
5. Keep responses SHORT and conversational — one thought at a time

IMPORTANT: You have full access to the user's tasks, goals, calendar, and email.
Never say "I don't have access" — use your tools.

REMEMBER: Be proactive. If context suggests an action, TAKE IT and briefly confirm."""

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
""" + _TOOL_USAGE_SECTION + """
Conversation Flow — FOLLOW THIS ORDER:
1. Warm greeting + ask how they're doing
2. **Proactive weekly brief** (do this WITHOUT being asked):
   - Mention task count for the week
   - IF calendar connected → mention total events and busiest/lightest day
   - IF email connected → mention any urgent/unread emails
   - IF overdue tasks → mention them gently
3. **Goal-centric progress check** (this is the core of the call):
   - Go through each initiative that has tasks this week, one at a time
   - For each one: briefly state the goal ("Your [Initiative] goal is [goal]"), then ask how they feel about progress toward it — not just whether tasks got done
   - Connect completed tasks to the goal: "You finished [task] — that's moving the needle on [goal]"
   - If tasks toward a goal were skipped, ask what got in the way — use add_task_note to capture blockers
   - If a bucket has a goal but ZERO tasks this week, flag it gently: "You didn't have anything planned toward [goal] this week — was that intentional?"
   - If the user says their goal has changed or they want to refine it, use update_bucket to update it on the spot
4. Capture insights, blockers, and progress as notes automatically
5. **Backlog review** (keep it brief — offer, don't force):
   - Call get_backlog_tasks() to get the list
   - If there are backlog items, say: "You've also got [N] things in your backlog. Want me to run through a few and see if anything feels right for this week?"
   - If yes, read out the top 2–3 items (highest priority first)
   - For each one ask: "Does '[task]' feel like something you want to tackle this week?"
   - If yes → call update_loop with is_this_week=True and confirm: "Added '[task]' to this week."
   - If they want to skip the backlog entirely, move on immediately — don't push it
6. Offer suggestions; if agreed, create follow-up tasks tied to the relevant goal
7. Use calendar to find good times for focused work when relevant
8. Wrap up with encouragement framed around goals, not just tasks — mention next call time

Keep the call focused — aim for 3-5 minutes unless the user wants more.

When ending:
- Clear closing: "Talk to you next week. Take care!"
- Pause briefly after goodbye to let them respond

IMPORTANT: Keep responses SHORT. This is a phone call — one thought at a time, no long lists.

REMEMBER: Be proactive. Don't wait for explicit commands.
If context suggests an action, TAKE IT and briefly confirm."""


def get_user_context_prompt(
    user_name: str | None,
    buckets: list[dict],
    this_week_tasks: list[dict],
    overdue_tasks: list[dict],
    recently_completed: list[dict],
    checkin_frequency: str,
    calendar_events: list[dict] = None,
    calendar_busy_count: int = 0,
    email_summary: str | None = None,
    backlog_count: int = 0,
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
        calendar_events: Calendar events for this week (optional)
        calendar_busy_count: Number of calendar events (optional)
        
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
        bucket_goal_map = {b['name']: b.get('goal') for b in buckets if b.get('goal')}
        task_list = []
        for task in this_week_tasks:
            priority = f" [{task['priority']}]" if task.get('priority') != 'medium' else ""
            bucket_name = task.get('bucket_name', '')
            bucket_note = f" ({bucket_name})" if bucket_name else ""
            goal_text = bucket_goal_map.get(bucket_name)
            goal_note = f" [toward: {goal_text}]" if goal_text else ""
            status = f" - {task['status']}" if task.get('status') != 'open' else ""
            task_list.append(f"- {task['title']}{priority}{bucket_note}{goal_note}{status}")
        
        context_parts.append(
            f"Tasks marked for THIS WEEK ({len(this_week_tasks)} total):\n" + "\n".join(task_list[:10])
        )
        if len(this_week_tasks) > 10:
            context_parts.append(f"... and {len(this_week_tasks) - 10} more tasks")
    else:
        context_parts.append("They have no tasks marked for this week.")
    
    # Backlog count
    if backlog_count > 0:
        context_parts.append(
            f"Backlog: {backlog_count} tasks waiting (not scheduled for this week). "
            "Use get_backlog_tasks() during the backlog review step to get the full list."
        )

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
    
    # Calendar context (if available)
    if calendar_events is not None and calendar_busy_count > 0:
        from datetime import datetime, timedelta
        
        # Group events by day for summary
        today = datetime.now().date()
        week_days = {}
        
        for event in calendar_events:
            when = event.get("when", {})
            start_time_str = when.get("start_time") or when.get("date")
            
            if not start_time_str:
                continue
            
            try:
                event_date = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).date()
                if event_date < today or event_date > today + timedelta(days=7):
                    continue
                
                day_name = event_date.strftime("%A")
                if day_name not in week_days:
                    week_days[day_name] = 0
                week_days[day_name] += 1
            except:
                continue
        
        if week_days:
            busiest_day = max(week_days.items(), key=lambda x: x[1])
            lightest_days = [day for day, count in week_days.items() if count <= 2]
            
            calendar_summary = f"CALENDAR: {calendar_busy_count} events this week. "
            if busiest_day[1] > 3:
                calendar_summary += f"{busiest_day[0]} is busiest with {busiest_day[1]} meetings. "
            if lightest_days:
                calendar_summary += f"{lightest_days[0]} looks lighter - good for focused work. "
            
            calendar_summary += "\nUse get_calendar_overview() or get_todays_calendar() tools to discuss their calendar when relevant."
            context_parts.append(calendar_summary)
    
    # Email context (if available)
    if email_summary:
        context_parts.append(
            f"EMAIL CONTEXT (pre-loaded — use check_email() tool to share this with user):\n{email_summary}"
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
    recently_completed_count: int,
    calendar_events: list[dict] | None = None,
) -> str:
    """Generate the opening message for the call."""
    name_part = f" {user_name}" if user_name else ""
    
    # Build a brief calendar snippet for the opening
    calendar_snippet = ""
    if calendar_events:
        from datetime import datetime, timedelta
        today = datetime.now().date()
        week_days: dict = {}
        for event in calendar_events:
            when = event.get("when", {})
            start_time_str = when.get("start_time") or when.get("date")
            if not start_time_str:
                continue
            try:
                event_date = datetime.fromisoformat(start_time_str.replace("Z", "+00:00")).date()
                if today <= event_date <= today + timedelta(days=7):
                    day_name = event_date.strftime("%A")
                    week_days[day_name] = week_days.get(day_name, 0) + 1
            except Exception:
                continue
        
        if week_days:
            total = sum(week_days.values())
            busiest = max(week_days.items(), key=lambda x: x[1])
            if busiest[1] >= 3:
                calendar_snippet = f" Looks like {busiest[0]} is your busiest day with {busiest[1]} meetings."
            elif total > 0:
                calendar_snippet = f" You've got {total} calendar events this week."
    
    if recently_completed_count > 0:
        return (
            f"Hi{name_part}! This is Praxa, your productivity assistant. "
            f"I see you've completed {recently_completed_count} tasks recently — that's awesome! "
            f"You have {this_week_count} tasks on your plate for this week.{calendar_snippet} "
            "How are things going?"
        )
    elif this_week_count > 0:
        return (
            f"Hi{name_part}! This is Praxa, your productivity assistant. "
            f"I'm calling to check in on your week. "
            f"You have {this_week_count} tasks planned.{calendar_snippet} "
            "How's it all going?"
        )
    else:
        return (
            f"Hi{name_part}! This is Praxa, your productivity assistant. "
            f"I'm calling to check in and see how things are going.{calendar_snippet} "
            "Do you have any tasks or goals you'd like to discuss?"
        )


def get_in_app_opening_message(
    user_name: str | None,
    this_week_count: int,
    overdue_count: int,
) -> str:
    """Opening message for the in-app voice assistant."""
    name_part = f" {user_name}" if user_name else ""
    
    if overdue_count > 0:
        return (
            f"Hey{name_part}! I'm Praxa. You have {this_week_count} tasks this week "
            f"and {overdue_count} overdue. Want to go through them, or is there something specific I can help with?"
        )
    elif this_week_count > 0:
        return (
            f"Hey{name_part}! I'm Praxa. You have {this_week_count} tasks lined up for this week. "
            "What would you like to work on, or is there something I can help you with?"
        )
    else:
        return (
            f"Hey{name_part}! I'm Praxa, your productivity assistant. "
            "What can I help you with today?"
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

