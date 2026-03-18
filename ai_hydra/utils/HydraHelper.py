# ai_hydra/utils/HydraHelper.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


# Helper function
def minutes_to_uptime(seconds: int):
    # Return a string like:
    # 45s
    # 7h 23m
    # 1d 7h 32m
    days, minutes = divmod(int(seconds), 86400)
    hours, minutes = divmod(minutes, 3600)
    minutes, seconds = divmod(minutes, 60)

    if days > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            minutes = f" {minutes}"
        if hours < 10:
            hours = f" {hours}"
        return f"{days}d {hours}h {minutes}m"

    elif hours > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            minutes = f" {minutes}"
        return f"{hours}h {minutes}m"

    elif minutes > 0:
        if seconds < 10:
            seconds = f" {seconds}"
        if minutes < 10:
            return f" {minutes}m {seconds}s"
        return f"{minutes}m {seconds}s"

    else:
        return f"{seconds}s"
