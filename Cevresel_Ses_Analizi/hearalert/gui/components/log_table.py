"""
HearAlert - Log Table (Tespit Gecmisi)
=========================================
Algilanan seslerin zaman damgali log tablosu.
Kritik tespitler kirmizi, normal tespitler yesil.
Maks 200 girdi, eski girdiler otomatik silinir.
"""

import customtkinter as ctk
from gui.styles import theme
import csv
from pathlib import Path


class LogTable(ctk.CTkScrollableFrame):
    """Tespit gecmisi tablosu."""

    def __init__(self, parent):
        super().__init__(parent, fg_color=theme.PANEL_BG,
                         corner_radius=8, height=180)

        self._entries = []
        self._max_entries = 200

        # Baslik satiri
        header = ctk.CTkFrame(self, fg_color=theme.CARD_BG, height=30,
                               corner_radius=4)
        header.pack(fill="x", pady=(0, 5), padx=2)
        header.pack_propagate(False)

        cols = [("Zaman", 80), ("Ses Sinifi", 140),
                ("Guven", 70), ("Durum", 100)]
        for text, width in cols:
            ctk.CTkLabel(header, text=text, font=("Consolas", 11, "bold"),
                         text_color=theme.TEXT_PRIMARY, width=width,
                         anchor="w").pack(side="left", padx=5)

    def add_entry(self, class_name: str, confidence: float,
                  is_critical: bool, timestamp: str):
        """Yeni girdi ekler."""
        # Maks girdi kontrolu
        if len(self._entries) >= self._max_entries:
            oldest = self._entries.pop(0)
            try:
                oldest.destroy()
            except Exception:
                pass

        idx = len(self._entries)
        bg = "#2a1525" if is_critical else (
            "#1e2a3e" if idx % 2 == 0 else "#1a2535")

        row = ctk.CTkFrame(self, fg_color=bg, height=28, corner_radius=2)
        row.pack(fill="x", pady=1, padx=2)
        row.pack_propagate(False)

        # Zaman
        ctk.CTkLabel(row, text=timestamp, font=theme.FONT_LOG,
                     text_color=theme.TEXT_SECONDARY, width=80,
                     anchor="w").pack(side="left", padx=5)

        # Sinif
        ctk.CTkLabel(row, text=class_name, font=theme.FONT_LOG,
                     text_color=theme.TEXT_PRIMARY, width=140,
                     anchor="w").pack(side="left", padx=5)

        # Guven
        ctk.CTkLabel(row, text=f"{confidence:.0%}", font=theme.FONT_LOG,
                     text_color=theme.TEXT_PRIMARY, width=70,
                     anchor="w").pack(side="left", padx=5)

        # Durum
        status_text = "!! KRITIK" if is_critical else "OK"
        status_color = theme.CRITICAL_RED if is_critical else theme.SAFE_GREEN
        ctk.CTkLabel(row, text=status_text, font=theme.FONT_LOG,
                     text_color=status_color, width=100,
                     anchor="w").pack(side="left", padx=5)

        self._entries.append(row)

        # Otomatik scroll
        self._parent_canvas.yview_moveto(1.0)

    def clear(self):
        """Tum girdileri temizle."""
        for entry in self._entries:
            try:
                entry.destroy()
            except Exception:
                pass
        self._entries.clear()

    def export_to_csv(self, file_path: Path):
        """Log'u CSV olarak disari aktar."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Class", "Confidence", "Status"])
