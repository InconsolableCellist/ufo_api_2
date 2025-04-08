import logging
from colorama import Fore, Back, Style

# Configure logging with custom formatter for colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }

    def format(self, record):
        log_message = super().format(record)
        
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        
        if "LLM response:" in log_message:
            # Highlight LLM responses in magenta
            parts = log_message.split("LLM response: '")
            if len(parts) > 1:
                response_part = parts[1].rsplit("'", 1)
                if len(response_part) > 1:
                    log_message = parts[0] + "LLM response: '" + Fore.MAGENTA + response_part[0] + Fore.RESET + "'" + response_part[1]
        
        # Highlight emotional states
        if "emotional state:" in log_message.lower():
            # Find the emotional state part and color it
            parts = log_message.split("emotional state:")
            if len(parts) > 1:
                state_part = parts[1].split("\n", 1)
                if len(state_part) > 1:
                    log_message = parts[0] + "emotional state:" + Fore.BLUE + state_part[0] + Fore.RESET + "\n" + state_part[1]
                else:
                    log_message = parts[0] + "emotional state:" + Fore.BLUE + state_part[0] + Fore.RESET
        
        return color + log_message + Style.RESET_ALL
