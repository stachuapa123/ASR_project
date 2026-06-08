import os, glob
from src.constants import Constants as C
from src.levenshtein import damerau_levenshtein_weighted, damerau_levenshtein_neighbour_aware
from src.parsers import wav_to_logmel
from src.evaluator import evaluate_word
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

from pathlib import Path

from src.parsers import PhonemeWindowDataset
from src.NeuralModel import CRNN
from src.trainers import train_model, evaluate_tm, load_checkpoint
from src.evaluator import evaluate_audio
from src.wordmaker import PHONEME_TO_LETTERS, levenshtein_distance, phonemes_to_text, parse_words, WLIST1000, proba_predict

from src.ctc.features import wav_path_to_logmel
from src.ctc.dataset import textgrid_to_phone_ids
from src.ctc.metrics import greedy_decode, decode_to_phones, compute_per
from dictionaries.dictmaker import load_word_list
from src.ctc.config import CTCConfig as Ct
from src.ctc.model import CTCModel


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "../trained_models/ctc_all_augmentations_45epochs.pt"

checkpoint = torch.load(checkpoint_path, map_location=device)

ctc_model = CTCModel().to(device)
ctc_model.load_state_dict(checkpoint["model_state_dict"])
ctc_model.eval()


CHECKPOINT_PATH = "../trained_models/BetterDataSoft.pth"

crnn_model = CRNN()
meta = load_checkpoint(CHECKPOINT_PATH, crnn_model, device=device)
crnn_model.eval()


import customtkinter as ctk
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import time


ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('dark-blue')


class DualModelRecognizer:
    def __init__(self, root, 
                 crnn_model, ctc_model,
                 device, word_list,
                 evaluate_audio_fn, proba_predict_fn, phonemes_to_text_fn,
                 wav_to_logmel_fn, greedy_decode_fn, decode_to_phones_fn,
                 distance_crnn_fn, distance_fn,
                 sample_rate=16000, max_duration=10.0):
        self.root = root
        
        self.crnn_model = crnn_model
        self.ctc_model = ctc_model
        self.device = device
        self.word_list = word_list
        self.sample_rate = sample_rate
        self.max_duration = max_duration         # bezpiecznik — max długość nagrania
        
        self.evaluate_audio = evaluate_audio_fn
        self.proba_predict = proba_predict_fn
        self.phonemes_to_text = phonemes_to_text_fn
        
        self.wav_to_logmel = wav_to_logmel_fn
        self.greedy_decode = greedy_decode_fn
        self.decode_to_phones = decode_to_phones_fn
        
        self.distance_fn = distance_fn
        self.distance_crnn_fn = distance_crnn_fn
        
        self.recording_path = '../recordings/output.wav'
        os.makedirs(os.path.dirname(self.recording_path), exist_ok=True)
        
        # NEW — stan nagrywania
        self.is_recording = False
        self.recorded_chunks = []
        self.stream = None
        self.recording_start_time = None
        
        self._build_ui()
    
    def _build_ui(self):
        self.root.title('Phoneme Recognizer — Dual Model')
        self.root.geometry('1400x950')
        
        main_frame = ctk.CTkFrame(self.root, fg_color='transparent')
        main_frame.pack(fill='both', expand=True, padx=40, pady=30)
        
        # ── Tytuł ──
        title = ctk.CTkLabel(
            main_frame, text='🎙️  Phoneme Recognizer',
            font=ctk.CTkFont(family='DejaVu Sans', size=32, weight='bold'),
        )
        title.pack(pady=(0, 5))
        
        subtitle = ctk.CTkLabel(
            main_frame, text='Porównanie modeli CRNN vs CTC',
            font=ctk.CTkFont(family='DejaVu Sans', size=14, slant='italic'),
            text_color='gray60',
        )
        subtitle.pack(pady=(0, 20))
        
        # ── Status ──
        self.status_label = ctk.CTkLabel(
            main_frame, text='● Gotowy',
            font=ctk.CTkFont(family='DejaVu Sans', size=20, weight='bold'),
            text_color='#66cc66',
        )
        self.status_label.pack(pady=(0, 15))
        
        # ── Przycisk START/STOP (zmienia się dynamicznie) ──
        self.record_button = ctk.CTkButton(
            main_frame, text='🎤  START',
            command=self._toggle_recording,
            font=ctk.CTkFont(family='DejaVu Sans', size=26, weight='bold'),
            width=350, height=80,
            corner_radius=15,
            fg_color='#cc4444', hover_color='#aa3333',
        )
        self.record_button.pack(pady=10)
        
        # ── Timer nagrywania ──
        self.timer_label = ctk.CTkLabel(
            main_frame, text='',
            font=ctk.CTkFont(family='DejaVu Sans Mono', size=24, weight='bold'),
            text_color='#888888',
        )
        self.timer_label.pack(pady=5)
        
        # ── Pulsujący wskaźnik nagrywania ──
        self.recording_indicator = ctk.CTkLabel(
            main_frame, text='',
            font=ctk.CTkFont(family='DejaVu Sans', size=14),
        )
        self.recording_indicator.pack(pady=5)
        
        # ── DWIE KOLUMNY na wyniki ──
        columns_frame = ctk.CTkFrame(main_frame, fg_color='transparent')
        columns_frame.pack(fill='both', expand=True, pady=20)
        
        self.crnn_column = self._make_model_column(
            columns_frame, 'CRNN (per-window)', '#3b6bcc',
        )
        self.crnn_column['frame'].pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.ctc_column = self._make_model_column(
            columns_frame, 'CTC (sequence)', '#cc6b3b',
        )
        self.ctc_column['frame'].pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Hotkeys
        self.root.bind('<space>', lambda e: self._toggle_recording())
        self.root.bind('<Return>', lambda e: self._toggle_recording())
    
    def _make_model_column(self, parent, title, accent_color):
        frame = ctk.CTkFrame(parent, fg_color='#252525', corner_radius=15)
        inner = ctk.CTkFrame(frame, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            inner, text=title,
            font=ctk.CTkFont(family='DejaVu Sans', size=20, weight='bold'),
            text_color=accent_color,
        ).pack(pady=(0, 15))
        
        ctk.CTkLabel(
            inner, text='Fonemy:',
            font=ctk.CTkFont(family='DejaVu Sans', size=13, weight='bold'),
            text_color='gray60', anchor='w',
        ).pack(fill='x', pady=(0, 3))
        
        phonemes_label = ctk.CTkLabel(
            inner, text='—',
            font=ctk.CTkFont(family='DejaVu Sans Mono', size=16, weight='bold'),
            fg_color='#1a1a1a', corner_radius=8,
            height=50, anchor='w',
        )
        phonemes_label.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            inner, text='Transkrypcja:',
            font=ctk.CTkFont(family='DejaVu Sans', size=13, weight='bold'),
            text_color='gray60', anchor='w',
        ).pack(fill='x', pady=(0, 3))
        
        text_label = ctk.CTkLabel(
            inner, text='—',
            font=ctk.CTkFont(family='DejaVu Sans', size=18),
            fg_color='#1a1a1a', corner_radius=8,
            height=50, anchor='w',
        )
        text_label.pack(fill='x', pady=(0, 10))
        
        ctk.CTkLabel(
            inner, text='💡  Najlepsze dopasowanie:',
            font=ctk.CTkFont(family='DejaVu Sans', size=13, weight='bold'),
            text_color='gray60', anchor='w',
        ).pack(fill='x', pady=(0, 3))
        
        match_label = ctk.CTkLabel(
            inner, text='—',
            font=ctk.CTkFont(family='DejaVu Sans', size=32, weight='bold'),
            fg_color=accent_color, corner_radius=12,
            height=90, text_color='white',
        )
        match_label.pack(fill='x', pady=(0, 5))
        
        distance_label = ctk.CTkLabel(
            inner, text='',
            font=ctk.CTkFont(family='DejaVu Sans', size=11, slant='italic'),
            text_color='gray60', anchor='e',
        )
        distance_label.pack(fill='x')
        
        return {
            'frame': frame,
            'phonemes': phonemes_label,
            'text': text_label,
            'match': match_label,
            'distance': distance_label,
        }
    
    # ─── NAGRYWANIE — TOGGLE ──────────────────────────────────────────
    def _toggle_recording(self):
        """Przełącza między start a stop nagrywania."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Rozpoczyna nagrywanie używając InputStream (asynchroniczny)."""
        if self.is_recording:
            return
        
        # Reset wyników
        for col in [self.crnn_column, self.ctc_column]:
            col['phonemes'].configure(text='—')
            col['text'].configure(text='—')
            col['match'].configure(text='—')
            col['distance'].configure(text='')
        
        self.is_recording = True
        self.recorded_chunks = []
        self.recording_start_time = time.time()
        
        # UI feedback
        self.record_button.configure(
            text='⏹  STOP',
            fg_color='#666666', hover_color='#444444',
        )
        self.status_label.configure(text='🔴 Nagrywanie...', text_color='#ff6666')
        
        # Callback wywoływany dla każdego chunk audio
        def audio_callback(indata, frames, time_info, status):
            if self.is_recording:
                self.recorded_chunks.append(indata.copy())
        
        # Otwórz stream w trybie callback (asynchroniczny)
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=1024,
            )
            self.stream.start()
            
            # Uruchom timer (uaktualnia UI co 100ms)
            self._update_timer()
            self._pulse_indicator()
        except Exception as e:
            self.is_recording = False
            self._show_error(f'Nie można uruchomić nagrywania: {e}')
    
    def _stop_recording(self):
        """Zatrzymuje nagrywanie i uruchamia przetwarzanie."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Zamknij stream
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Reset UI nagrywania
        self.record_button.configure(
            text='🎤  START', state='disabled',
            fg_color='#444444',
        )
        self.timer_label.configure(text='')
        self.recording_indicator.configure(text='')
        self.status_label.configure(text='⚙️ Przetwarzanie...', text_color='#ffaa66')
        
        # Sprawdź czy mamy jakieś audio
        if not self.recorded_chunks:
            self._show_error('Nagranie jest puste!')
            return
        
        # Zapisz audio
        recording = np.concatenate(self.recorded_chunks, axis=0)
        duration_sec = len(recording) / self.sample_rate
        
        if duration_sec < 0.3:
            self._show_error(f'Nagranie za krótkie ({duration_sec:.1f}s, min 0.3s)')
            return
        
        write(self.recording_path, self.sample_rate, recording)
        
        # Uruchom processing w wątku tła
        threading.Thread(target=self._process_recording, daemon=True).start()
    
    def _update_timer(self):
        """Aktualizuje wyświetlany czas nagrywania co 100ms."""
        if not self.is_recording:
            return
        
        elapsed = time.time() - self.recording_start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        self.timer_label.configure(text=f'{minutes:02d}:{seconds:05.2f}')
        
        # Bezpiecznik — auto-stop po max_duration
        if elapsed >= self.max_duration:
            self.status_label.configure(
                text=f'⚠️ Max długość ({self.max_duration}s), auto-stop',
                text_color='#ffaa66',
            )
            self._stop_recording()
            return
        
        self.root.after(100, self._update_timer)
    
    def _pulse_indicator(self):
        """Pulsujący wskaźnik nagrywania (●○●○...)."""
        if not self.is_recording:
            self.recording_indicator.configure(text='')
            return
        
        current = self.recording_indicator.cget('text')
        if current == '🔴 REC':
            self.recording_indicator.configure(text='⚪ REC', text_color='#888888')
        else:
            self.recording_indicator.configure(text='🔴 REC', text_color='#ff6666')
        
        self.root.after(500, self._pulse_indicator)
    
    # ─── PRZETWARZANIE ─────────────────────────────────────────────────
    def _process_recording(self):
        """Pełen pipeline w wątku tła."""
        try:
            self.root.after(0, self._update_status, '⚙️ CRNN predicting...', '#ffaa66')
            
            crnn_results = self._predict_crnn()
            self.root.after(0, self._show_crnn_results, *crnn_results)
            
            self.root.after(0, self._update_status, '⚙️ CTC predicting...', '#ffaa66')
            
            ctc_results = self._predict_ctc()
            self.root.after(0, self._show_ctc_results, *ctc_results)
            
            self.root.after(0, self._mark_done)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, self._show_error, str(e))
    
    def _predict_crnn(self):
        result = self.evaluate_audio(
            self.recording_path,
            model=self.crnn_model,
            device=self.device,
            show_per_window=False,
            top_k=4,
        )
        
        phonemes = self.proba_predict(result, p=0.67, longer_reg=True, aeo_reg=True)
        text = self.phonemes_to_text(phonemes, after_silence=False)
        
        output = self.word_list[0]
        min_dist = float('inf')
        for word in self.word_list:
            dist = self.distance_crnn_fn(word, phonemes)
            if dist < min_dist:
                min_dist = dist
                output = word
        
        return phonemes, text, output, min_dist
    
    def _predict_ctc(self):
        import torch
        
        mel = self.wav_to_logmel(self.recording_path)
        mel = mel.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.ctc_model(mel)
        
        pred_ids = self.greedy_decode(logits.cpu())[0]

        decoded_phonemes = [C.IDX2LABEL.get(i, "<UNK>") for i in pred_ids]
        decoded_text = phonemes_to_text(decoded_phonemes, after_silence=False)
        result = decoded_text.replace(" ", "").strip()
        
        #decoded = self.decode_to_phones(pred_ids)
        #decoded_text = self.phonemes_to_text(decoded, after_silence=False)
        #phonemes = [str(ph) for ph in decoded_text.split()]
        #result = decoded_text.replace("sil", "").replace(" ", "")
        output = self.word_list[0]
        min_dist = float('inf')
        for word in self.word_list:
            dist = self.distance_fn(word, result)
            if dist < min_dist:
                min_dist = dist
                output = word
        
        return decoded_phonemes, result, output, min_dist
    
    # ─── UI UPDATES ────────────────────────────────────────────────────
    def _update_status(self, text, color):
        self.status_label.configure(text=text, text_color=color)
    
    def _show_crnn_results(self, phonemes, text, match, distance):
        self.crnn_column['phonemes'].configure(text=' '.join(phonemes))
        self.crnn_column['text'].configure(text=text)
        self.crnn_column['match'].configure(text=match.upper())
        self.crnn_column['distance'].configure(text=f'distance: {distance:.2f}')
    
    def _show_ctc_results(self, phonemes, text, match, distance):
        self.ctc_column['phonemes'].configure(text=' '.join(phonemes))
        self.ctc_column['text'].configure(text=text)
        self.ctc_column['match'].configure(text=match.upper())
        self.ctc_column['distance'].configure(text=f'distance: {distance:.2f}')
    
    def _mark_done(self):
        self.status_label.configure(text='● Gotowy', text_color='#66cc66')
        self.record_button.configure(
            state='normal', text='🎤  START',
            fg_color='#cc4444', hover_color='#aa3333',
        )
        self.timer_label.configure(text='')
        self.recording_indicator.configure(text='')
    
    def _show_error(self, error_msg):
        self.status_label.configure(
            text=f'❌ Błąd: {error_msg[:80]}',
            text_color='#ff6666',
        )
        self.record_button.configure(
            state='normal', text='🎤  START',
            fg_color='#cc4444', hover_color='#aa3333',
        )
        self.timer_label.configure(text='')
        self.recording_indicator.configure(text='')


# ─── Użycie ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = ctk.CTk()
    
    app = DualModelRecognizer(
        root,
        crnn_model=crnn_model,
        ctc_model=ctc_model,
        device=device,
        word_list=load_word_list('../dictionaries/sample_words.txt'),
        
        evaluate_audio_fn=evaluate_audio,
        proba_predict_fn=proba_predict,
        phonemes_to_text_fn=phonemes_to_text,
        
        wav_to_logmel_fn=wav_path_to_logmel,
        greedy_decode_fn=greedy_decode,
        decode_to_phones_fn=decode_to_phones,
        
        distance_crnn_fn=damerau_levenshtein_weighted,
        distance_fn=damerau_levenshtein_neighbour_aware,
        
        sample_rate=C.SAMPLE_RATE,
        max_duration=10.0,    # bezpiecznik — max 10s nagrania
    )
    root.mainloop()