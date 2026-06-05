"""
HearAlert - Alert Card (Bildirim Karti)
==========================================
Ses algilandiginda gosterilen bildirim karti.
Kritik: kirmizi arka plan, Normal: mavi arka plan.
6 saniye sonra otomatik kapanir.
"""

import customtkinter as ctk
from gui.styles import theme


class AlertCard(ctk.CTkFrame):
    """Canli bildirim karti bileeni."""

    def __init__(self, parent, class_name: str, confidence: float,
                 is_critical: bool, timestamp: str):
        bg = theme.CRITICAL_RED if is_critical else theme.CARD_BG
        super().__init__(parent, fg_color=bg, corner_radius=theme.CARD_BORDER_RADIUS,
                         height=90)
        self.pack_propagate(False)

        icon = theme.CLASS_ICONS.get(class_name, "[?]")
        status = "!! KRITIK" if is_critical else "OK"

        # Icerik frame
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=theme.CARD_PADDING,
                     pady=theme.PAD_SM)

        # Ust satir: icon + sinif adi
        top = ctk.CTkFrame(content, fg_color="transparent")
        top.pack(fill="x")

        ctk.CTkLabel(top, text=f"{icon}  {class_name.upper()}",
                     font=theme.FONT_CARD_CLASS, text_color="white",
                     anchor="w").pack(side="left")

        ctk.CTkLabel(top, text=status,
                     font=theme.FONT_SMALL,
                     text_color=theme.WARNING_YELLOW if is_critical else theme.SAFE_GREEN,
                     anchor="e").pack(side="right")

        # Alt satir: guven + zaman
        bottom = ctk.CTkFrame(content, fg_color="transparent")
        bottom.pack(fill="x", pady=(4, 0))

        ctk.CTkLabel(bottom, text=f"Confidence: {confidence:.1%}",
                     font=theme.FONT_CARD_CONF, text_color=theme.TEXT_PRIMARY,
                     anchor="w").pack(side="left")

        ctk.CTkLabel(bottom, text=timestamp,
                     font=theme.FONT_SMALL, text_color=theme.TEXT_SECONDARY,
                     anchor="e").pack(side="right")

        # Otomatik kapanma
        self.after(theme.CARD_DISPLAY_DURATION_MS, self._dismiss)

    def _dismiss(self):
        """Karti kaldir."""
        try:
            self.destroy()
        except Exception:
            pass
