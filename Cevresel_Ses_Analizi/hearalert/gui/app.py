"""
HearAlert - Ana CustomTkinter GUI Uygulamasi
================================================
1100x700 pencere, dark mode, 2 sutunlu layout.
Sol: Alert card alani + flash overlay
Sag: Confidence panel
Alt: Detection log tablosu
GUI event loop: 100ms aralikla gui_queue kontrolu.
"""

import customtkinter as ctk
import queue
import sys
from pathlib import Path
from datetime import datetime

# Proje kokunu ayarla
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui.styles import theme
from gui.components.flash_overlay import FlashOverlay
from gui.components.alert_card import AlertCard
from gui.components.confidence_panel import ConfidencePanel
from gui.components.log_table import LogTable
from src.config import config


class HearAlertApp(ctk.CTk):
    """HearAlert ana uygulama penceresi."""

    def __init__(self):
        super().__init__()

        # -- Pencere ayarlari --
        self.title("HearAlert - Sound Event Recognition")
        self.geometry("1100x700")
        self.minsize(900, 600)
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=theme.DARK_BG)

        # -- Durum degiskenleri --
        self._running = False
        self._model_type = "cnn"
        self._predictor = None
        self._listener = None
        self._gui_queue = queue.Queue(maxsize=50)
        self._alert_cards = []

        # -- UI olustur --
        self._build_top_bar()
        self._build_main_content()
        self._build_bottom_log()
        self._build_status_bar()

        # -- Flash overlay --
        self.flash_overlay = FlashOverlay(self)

        # -- Pencere kapanma --
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_top_bar(self):
        """Ust bar: baslik + kontroller."""
        top = ctk.CTkFrame(self, fg_color=theme.PANEL_BG, height=60,
                            corner_radius=0)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)

        # Sol: baslik + canli gostergesi
        left = ctk.CTkFrame(top, fg_color="transparent")
        left.pack(side="left", padx=theme.PAD_LG)

        ctk.CTkLabel(left, text="HearAlert",
                     font=theme.FONT_TITLE,
                     text_color=theme.SAFE_GREEN).pack(side="left")

        self._live_dot = ctk.CTkLabel(left, text="  [STOPPED]",
                                       font=theme.FONT_SMALL,
                                       text_color=theme.TEXT_SECONDARY)
        self._live_dot.pack(side="left", padx=10)

        # Sag: butonlar
        right = ctk.CTkFrame(top, fg_color="transparent")
        right.pack(side="right", padx=theme.PAD_LG)

        # Model secimi
        self._model_var = ctk.StringVar(value="CNN")
        model_menu = ctk.CTkOptionMenu(
            right, values=["MLP", "CNN"], variable=self._model_var,
            command=self._on_model_change, width=80,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG)
        model_menu.pack(side="left", padx=5)

        # Flash switch
        self._flash_var = ctk.BooleanVar(value=False)
        flash_switch = ctk.CTkSwitch(
            right, text="Flash", variable=self._flash_var, width=60,
            progress_color=theme.CRITICAL_RED)
        flash_switch.pack(side="left", padx=10)

        # Mic secimi
        ctk.CTkButton(right, text="Mic", width=50,
                      fg_color=theme.CARD_BG, hover_color=theme.PANEL_BG,
                      command=self._show_mic_dialog).pack(side="left", padx=5)

        # Stop
        ctk.CTkButton(right, text="Stop", width=70,
                      fg_color=theme.CRITICAL_DARK,
                      hover_color=theme.CRITICAL_RED,
                      command=self._stop_listening).pack(side="left", padx=5)

        # Start
        ctk.CTkButton(right, text="Start", width=70,
                      fg_color="#0a7c4a", hover_color="#0d9b5f",
                      command=self._start_listening).pack(side="left", padx=5)

    def _build_main_content(self):
        """Ana icerik: sol alert + sag confidence."""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=theme.PAD_MD,
                  pady=theme.PAD_MD)

        # Sol: alert card alani (%60)
        self._alert_area = ctk.CTkScrollableFrame(
            main, fg_color=theme.DARK_BG, corner_radius=8,
            label_text="ALERTS", label_font=("Segoe UI", 14, "bold"),
            label_text_color=theme.TEXT_PRIMARY)
        self._alert_area.pack(side="left", fill="both", expand=True,
                               padx=(0, theme.PAD_SM))

        # Sag: confidence panel (%40)
        self._confidence_panel = ConfidencePanel(main)
        self._confidence_panel.pack(side="right", fill="both",
                                     padx=(theme.PAD_SM, 0), ipadx=10)
        self._confidence_panel.configure(width=350)

    def _build_bottom_log(self):
        """Alt: detection log tablosu."""
        self._log_table = LogTable(self)
        self._log_table.pack(fill="x", padx=theme.PAD_MD,
                              pady=(0, theme.PAD_SM))

    def _build_status_bar(self):
        """Status bar."""
        self._status_bar = ctk.CTkLabel(
            self, text="Ready | Model: CNN | SR: 22050 Hz",
            font=theme.FONT_STATUS, text_color=theme.TEXT_SECONDARY,
            fg_color=theme.PANEL_BG, height=25, anchor="w")
        self._status_bar.pack(fill="x", padx=0, pady=0)

    def _start_listening(self):
        """Dinlemeyi baslatir."""
        if self._running:
            return

        # Model yukle
        from src.inference.predictor import SoundPredictor
        from src.inference.realtime_listener import RealTimeListener

        model_type = self._model_var.get().lower()
        self._predictor = SoundPredictor(model_type=model_type)

        if not self._predictor.load_model():
            self._show_error("Model yuklenemedi! Once egitim yapin.")
            return

        self._listener = RealTimeListener(
            predictor=self._predictor,
            gui_queue=self._gui_queue
        )

        if self._listener.start():
            self._running = True
            self._live_dot.configure(text="  [LIVE]",
                                      text_color=theme.SAFE_GREEN)
            self._status_bar.configure(
                text=f"Listening... | Model: {model_type.upper()} | SR: {config.SAMPLE_RATE} Hz")
            self._poll_gui_queue()
        else:
            self._show_error("Mikrofon baslatilamadi!")

    def _stop_listening(self):
        """Dinlemeyi durdurur."""
        self._running = False
        if self._listener:
            self._listener.stop()
        self._live_dot.configure(text="  [STOPPED]",
                                  text_color=theme.TEXT_SECONDARY)
        self._status_bar.configure(text="Stopped | Ready")

    def _poll_gui_queue(self):
        """GUI event queue kontrolu (100ms aralikla)."""
        if not self._running:
            return

        try:
            while True:
                event = self._gui_queue.get_nowait()
                try:
                    self._handle_event(event)
                except Exception as e:
                    print(f"Error handling GUI event: {e}")
        except queue.Empty:
            pass
        finally:
            if self._running:
                self.after(config.GUI_POLL_INTERVAL_MS, self._poll_gui_queue)

    def _handle_event(self, event: dict):
        """Inference sonucunu isler."""
        # Confidence panel her zaman guncellenir
        if "all_probs" in event:
            self._confidence_panel.update_confidence(event["all_probs"])

        # Tespit varsa alert + log + flash
        if event.get("detected", False):
            cls = event["class"]
            conf = event["confidence"]
            is_crit = event["is_critical"]
            ts = event["timestamp"]

            # Flash
            if self._flash_var.get():
                self.flash_overlay.trigger_flash(is_crit)

            # Alert card
            card = AlertCard(self._alert_area, cls, conf, is_crit, ts)
            card.pack(fill="x", pady=3, padx=5)
            self._alert_cards.append(card)

            # Maks 4 kart
            while len(self._alert_cards) > 4:
                old = self._alert_cards.pop(0)
                try:
                    old.destroy()
                except Exception:
                    pass

            # Log
            self._log_table.add_entry(cls, conf, is_crit, ts)

            # Status bar guncelle
            ms = event.get("inference_ms", 0)
            self._status_bar.configure(
                text=f"Detected: {cls} ({conf:.0%}) | Inference: {ms:.1f}ms")

    def _on_model_change(self, value):
        """Model degistiginde yeniden baslat."""
        if self._running:
            self._stop_listening()
            self.after(500, self._start_listening)

    def _show_mic_dialog(self):
        """Mikrofon secim penceresi."""
        from src.inference.realtime_listener import RealTimeListener

        dialog = ctk.CTkToplevel(self)
        dialog.title("Mikrofon Secimi")
        dialog.geometry("400x300")
        dialog.configure(fg_color=theme.DARK_BG)
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Ses Giris Cihazlari",
                     font=("Segoe UI", 16, "bold"),
                     text_color=theme.TEXT_PRIMARY).pack(pady=10)

        devices = RealTimeListener.list_audio_devices()
        if not devices:
            ctk.CTkLabel(dialog, text="Cihaz bulunamadi!",
                         text_color=theme.CRITICAL_RED).pack(pady=20)
            return

        for dev in devices:
            btn = ctk.CTkButton(
                dialog,
                text=f"[{dev['index']}] {dev['name']}",
                fg_color=theme.CARD_BG, hover_color=theme.PANEL_BG,
                command=lambda idx=dev['index']: self._select_mic(idx, dialog))
            btn.pack(fill="x", padx=20, pady=3)

    def _select_mic(self, device_index: int, dialog):
        """Mikrofon sec ve kapat."""
        config.MIC_DEVICE_INDEX = device_index
        dialog.destroy()
        if self._running:
            self._stop_listening()
            self.after(500, self._start_listening)

    def _show_error(self, message: str):
        """Hata mesaji goster."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Hata")
        dialog.geometry("350x150")
        dialog.configure(fg_color=theme.DARK_BG)
        dialog.transient(self)

        ctk.CTkLabel(dialog, text=message, text_color=theme.CRITICAL_RED,
                     font=theme.FONT_LABEL, wraplength=300).pack(pady=20)
        ctk.CTkButton(dialog, text="Tamam",
                      command=dialog.destroy).pack(pady=10)

    def _on_close(self):
        """Pencere kapanirken temizlik."""
        self._stop_listening()
        self.destroy()


def launch_gui():
    """GUI'yi baslatir."""
    app = HearAlertApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
