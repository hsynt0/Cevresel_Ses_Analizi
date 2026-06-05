"""
HearAlert - Confidence Panel
================================
Softmax olasiliklari her inference cikriminda guncellenir.
"""

import customtkinter as ctk
from gui.styles import theme
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config


class ConfidencePanel(ctk.CTkScrollableFrame):
    """Tum siniflar icin canli confidence bar paneli."""

    def __init__(self, parent):
        super().__init__(parent, fg_color=theme.PANEL_BG,
                         corner_radius=8, label_text="CONFIDENCE",
                         label_font=("Segoe UI", 14, "bold"),
                         label_text_color=theme.TEXT_PRIMARY)

        self._bars = {}
        self._labels = {}
        self._pct_labels = {}

        for cls in config.CLASSES:
            row = ctk.CTkFrame(self, fg_color="transparent", height=28)
            row.pack(fill="x", pady=2, padx=5)
            row.pack_propagate(False)

            # Sinif adi
            lbl = ctk.CTkLabel(row, text=cls.replace("_", " "),
                               font=theme.FONT_SMALL,
                               text_color=theme.TEXT_SECONDARY,
                               width=110, anchor="w")
            lbl.pack(side="left")
            self._labels[cls] = lbl

            # Bar arka plan
            bar_bg = ctk.CTkFrame(row, fg_color="#2a2a4e", height=16,
                                   corner_radius=4)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(5, 5))
            bar_bg.pack_propagate(False)

            # Bar dolgu
            bar_fill = ctk.CTkFrame(bar_bg, fg_color=theme.CONFIDENCE_BAR_FILL,
                                     height=16, width=0, corner_radius=4)
            bar_fill.place(x=0, y=0, relheight=1)
            self._bars[cls] = (bar_bg, bar_fill)

            # Yuzde
            pct = ctk.CTkLabel(row, text="0%", font=theme.FONT_SMALL,
                               text_color=theme.TEXT_SECONDARY, width=45,
                               anchor="e")
            pct.pack(side="right")
            self._pct_labels[cls] = pct

    def update_confidence(self, predictions: dict):
        """Tum cubukları gunceller.

        Args:
            predictions: {sinif_adi: olasilik} sozlugu (tum siniflar).
        """
        if not predictions:
            return

        top_class = max(predictions, key=predictions.get)

        for cls in config.CLASSES:
            prob = predictions.get(cls, 0.0)
            pct = int(prob * 100)

            bar_bg, bar_fill = self._bars[cls]
            max_width = bar_bg.winfo_width()
            if max_width < 10:
                max_width = 200
            target_w = max(1, int(prob * max_width))
            bar_fill.configure(width=target_w)
            bar_fill.place(x=0, y=0, relheight=1)

            # Renk: guvenlik-kritik + yuksek olasilik = kirmizi
            is_critical = cls in config.SAFETY_CRITICAL_CLASSES
            if is_critical and prob > 0.5:
                bar_fill.configure(fg_color=theme.CONFIDENCE_BAR_CRITICAL)
            else:
                bar_fill.configure(fg_color=theme.CONFIDENCE_BAR_FILL)

            self._pct_labels[cls].configure(text=f"{pct}%")

            # En yuksek sinifi vurgula
            if cls == top_class:
                self._labels[cls].configure(text_color=theme.TEXT_PRIMARY,
                                             font=("Segoe UI", 10, "bold"))
            else:
                self._labels[cls].configure(text_color=theme.TEXT_SECONDARY,
                                             font=theme.FONT_SMALL)
