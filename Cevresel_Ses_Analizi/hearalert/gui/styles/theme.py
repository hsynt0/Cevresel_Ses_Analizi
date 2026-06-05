"""
HearAlert - GUI Theme & Style Constants
==========================================
Tum renk paleti, font tanimlari ve spacing sabitleri.
"""

# -- COLORS --
DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
CARD_BG = "#0f3460"
CRITICAL_RED = "#e94560"
CRITICAL_DARK = "#c73652"
SAFE_GREEN = "#00b4d8"
WARNING_YELLOW = "#ffd60a"
TEXT_PRIMARY = "#eaeaea"
TEXT_SECONDARY = "#a0a0b0"
CONFIDENCE_BAR_FILL = "#00b4d8"
CONFIDENCE_BAR_CRITICAL = "#e94560"
ACCENT_PURPLE = "#7b2d8e"

# -- FONTS --
FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_CARD_CLASS = ("Segoe UI", 18, "bold")
FONT_CARD_CONF = ("Segoe UI", 13)
FONT_LOG = ("Consolas", 11)
FONT_LABEL = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 10)
FONT_STATUS = ("Consolas", 10)

# -- SPACING --
CARD_BORDER_RADIUS = 12
CARD_PADDING = 16
PAD_SM = 5
PAD_MD = 10
PAD_LG = 20

# -- ANIMATION --
FLASH_DURATION_MS = 120
FLASH_COUNT = 5
CARD_DISPLAY_DURATION_MS = 6000

# -- CLASS EMOJIS (ASCII-safe) --
CLASS_ICONS = {
    "alarm_clock": "[ALARM]",
    "crying_baby": "[BABY]",
    "glass_breaking": "[GLASS]",
    "dog_bark": "[DOG]",
    "siren": "[SIREN]",
}
