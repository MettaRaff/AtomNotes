# pyright: ignore[reportMissingImports]
import sounddevice as sd
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import tkinter as tk
from tkinter import ttk
from collections import deque
from collections import defaultdict
from threading import Thread, Event

# Конфигурация
DEVICE_ID = 1             # Ваш device_id
FS = 44100                # Частота дискретизации
BLOCK_SIZE = 2048         # Размер блока для БПФ
COM_PORT = 'COM6'         # Замените на ваш порт
BAUDRATE = 115200
CHANNELS = 1              # Количество каналов
BUFFER_SIZE = 1           # Размер буфера для сглаживания
FREQ_RANGE = (20, 20000)  # Диапазон частот
DIVIDE_A = 160.0 #Hz
DIVIDE_B = 2000.0 #Hz
SIDECHAIN_POWER = 1
SMOOTH_COEFF = 0.1

class Analyzer:
    def __init__(self, device_id, fs, block_size, channels, buffer_size, freq_range):
        print("It analyzer init def!")
        self.freq_bins = np.fft.fftfreq(BLOCK_SIZE, 1/FS)[:BLOCK_SIZE//2]
        self.data_queue = deque(maxlen=BUFFER_SIZE)
        self.stop_event = Event()
    
    def half_array(self, arr_inp):
        arr_out = []
        if len(arr_inp) <= 1:
            return arr_inp
        for i in range(0, len(arr_inp)-1, 2):
            half_val = (arr_inp[i] + arr_inp[i+1]) / 2
            arr_out.append(half_val)
        return arr_out
    
    def audio_callback(self, indata, frames, time, status):
    ###Обработка аудио данных###
        if status:
            print(f"Audio error: {status}")
    
        # Преобразование в моно
        #signal = indata[:,0] if indata.ndim > 1 else indata.flatten()
        # Конвертируем в моно, если нужно
        if indata.ndim > 1:
            signal = np.mean(indata, axis=1)
        else:
            signal = indata.flatten()
        signal = self.half_array(signal)
        # Оконная функция Ханна
        window = np.hanning(len(signal))
        windowed = signal * window
        # БПФ и преобразование в дБ
        fft = np.fft.fft(windowed)    
        magnitude = np.abs(fft) * 2 / np.sum(window)
        db_spectrum = 20 * np.log10(magnitude + 1e-12)
        self.data_queue.append(db_spectrum)

    def start_audio_stream(self):
        ###Запуск аудио потока###
        with sd.InputStream(
            device=DEVICE_ID,
            samplerate=FS,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            callback=self.audio_callback
        ):
            print("Аудиопоток запущен...")
            while not self.stop_event.is_set():
                sd.sleep(10)    

class GraphPlotter:
    def __init__(self, freq_bins, data_queue):
        print("It init graph plotter!")
        # Настройка графика
        self.avg_spectrum = []
        self.rgb_spectrum = []
        self.y_arr = []
        self.data_queue = data_queue
        self.freq_bins = freq_bins
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.semilogx(self.freq_bins, np.zeros_like(self.freq_bins))
        self.point, = self.ax.plot([], [], 'ro', markersize=8)
        #self.line2, = self.ax.semilogx(self.freq_bins, np.zeros_like(self.freq_bins), color='red')

        self.lines = self.ax.lines

        self.ax.set_xlim(FREQ_RANGE)
        self.ax.set_ylim(-80, 0)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Amplitude (dB)')
        self.ax.set_title('Real-Time Audio Spectrum')
        self.ax.grid(True, which='both', linestyle='--', alpha=0.5)
        self.fig.tight_layout()

    def update_plot(self, frame):
        """Обновление графика"""
        try:
            if self.data_queue:
                self.rgb_spectrum = self.avg_spectrum
                self.avg_spectrum = np.mean(self.data_queue, axis=0)                

                for i in range(len(self.lines)):
                    if(i == 0):  
                        self.lines[i].set_ydata(self.avg_spectrum)
                    elif(i == 1):
                        if(len(animModule.maxPointsX)>0):
                            self.point.set_data(animModule.maxPointsX,
                                                   animModule.maxPointsY)
                    #elif(i == 2):
                    #    if(len(self.rgb_spectrum)>0):
                    #        self.lines[i].set_ydata(self.rgb_spectrum)

            return self.point, self.line, #self.line2,
        except Exception as e:
            print(f"Ошибка обновления: {e}")
            return self.point, self.line, #self.line2,

class NoteDetector:
    def __init__(self, freqs):
        # Диапазон анализа (200-4000 Гц)
        self.min_freq = 200.0
        self.max_freq = 4000.0
        
        # Соответствие нот частотам (C4 = 261.63 Гц, A4 = 440 Гц и т.д.)
        self.note_freqs = self.create_note_table()
        
        # Цвета для нот (RGB значения)
        self.note_colors = {
            'C': (255, 0, 0),    # Красный
            'C#': (255, 128, 0), # Оранжевый
            'D': (255, 255, 0), # Желтый
            'D#': (128, 255, 0), # Лаймовый
            'E': (0, 255, 0),    # Зеленый
            'F': (0, 255, 128),  # Бирюзовый
            'F#': (0, 255, 255), # Голубой
            'G': (0, 128, 255), # Синий
            'G#': (0, 0, 255),   # Темно-синий
            'A': (128, 0, 255), # Фиолетовый
            'A#': (255, 0, 255), # Пурпурный
            'B': (255, 0, 128)  # Розовый
        }
        self.mask = []
        self.filtered_freqs = []
        self.filtered_amps = []
        self.note = ""
        self.sorted_notes = []
        for i in range(len(freqs)):
            if(freqs[i] >= self.min_freq and freqs[i] <= self.max_freq):
                self.mask.append(i)
        for i in range(len(self.mask)):
            self.filtered_freqs.append(freqs[self.mask[i]])

    def create_note_table(self):
        """Создает таблицу соответствия нот и частот"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_table = {}
        
        # Заполняем частоты для октав 3-8 (C3 = 130.81 Гц, A4 = 440 Гц и т.д.)
        for octave in range(3, 9):
            for i, note in enumerate(notes):
                freq = 440.0 * 2**((octave-4) + (i-9)/12.0)
                note_name = f"{note}{octave}"
                note_table[note_name] = freq
        return note_table

    def freq_to_note(self, freq):
        """Преобразует частоту в ближайшую ноту"""
        min_diff = float('inf')
        closest_note = None
        
        for note, note_freq in self.note_freqs.items():
            diff = abs(freq - note_freq)
            if diff < min_diff:
                min_diff = diff
                closest_note = note
        return closest_note.split('0')[0]  # Убираем октаву

    def find_peaks(self, freqs, amplitudes, threshold=-40):
        """Находит значимые пики в спектре"""
        peaks = []
        for i in range(1, len(amplitudes)-1):
            if amplitudes[i] > threshold and \
               amplitudes[i] > amplitudes[i-1] and \
               amplitudes[i] > amplitudes[i+1]:
                peaks.append((freqs[i], amplitudes[i]))
        return sorted(peaks, key=lambda x: x[1], reverse=True)[:5]  # Топ-5 пиков

    def detect_notes(self, amplitudes):
        ###Основная функция детекции нот###
        # Фильтрация по частотному диапазону        
        self.filtered_amps.clear()
        if(len(amplitudes)>0):
            for i in range(len(self.mask)):
                self.filtered_amps.append(amplitudes[self.mask[i]])                
        
        if len(self.filtered_amps) == 0:
            return []

        # Находим значимые пики
        self.peaks = self.find_peaks(self.filtered_freqs, self.filtered_amps)
        # Собираем ноты с весами
        if(len(self.peaks)>0):
            note_weights = defaultdict(float)
            for freq, amp in self.peaks:
                self.note = self.freq_to_note(freq)
                note_weights[self.note] += amp  # Вес ноты зависит от амплитуды
  
        # Выбираем 2 наиболее значимые ноты
            self.sorted_notes = sorted(note_weights.items(), key=lambda x: x[1], reverse=True)

        return [note for note, _ in self.sorted_notes[:2]]

    def get_note_color(self, notes):
        ###Возвращает цвет для списка нот###
        if not notes:
            return (0, 0, 0)  # Черный цвет при отсутствии нот
        
        # Смешиваем цвета для нескольких нот
        color = [0, 0, 0]
        for note in notes[:2]:  # Берем не более двух нот
            note = note[:-1]
            rgb = self.note_colors.get(note)            
            for i in range(3):
                color[i] += rgb[i]
        
        # Нормализация
        return tuple(min(255, int(c / len(notes))) for c in color)

class AppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Spectrum Analyzer")

        self.style = ttk.Style()
        self.style.configure("Custom.TFrame", background="lightcoral")

        self.control_frame = ttk.Frame(self.root, 
                                       padding=10, 
                                       width=200, 
                                       height=140,
                                       style="Custom.TFrame")
        self.control_frame.pack(pady=20, fill="both", expand=True)
        self.root.mainloop()

class AnimationModule:
    def __init__(self, freq_bins):
        self.freq_bins = freq_bins
        print("It init Anim Module!")
        self.divides = []
        self.avg_spectrum = []
        self.rgb_spectrum = []
        self.maxPointsX = []
        self.maxPointsY = []
        self.color_f = [0, 0, 0]
        #рассчитаем индексы разделительных частот    
        self.divides.append(self.searchGraphIndex(DIVIDE_A))
        self.divides.append(self.searchGraphIndex(DIVIDE_B))

        self.ser = serial.Serial(COM_PORT, BAUDRATE, timeout=1)

        self.kick = PointSidechain(60, 150, self.freq_bins)
        self.detector = NoteDetector(freq_bins)

    def __del__(self):
        self.ser.close()
        print("Закрытие СОМ порта...")

    def searchGraphIndex(self, freq):
        for i in range(0, BLOCK_SIZE//2, 1):
           if(freq <= fftAnalyzer.freq_bins[i]):
                return i
        return 0  
    
    def animClassic(self):
        if(len(self.avg_spectrum) > 0):
            bass = np.mean(self.avg_spectrum[4:self.divides[0]])    # 20-100 Гц
            mids = np.mean(self.avg_spectrum[self.divides[0]:self.divides[1]])  # 100-1000 Гц
            highs = np.mean(self.avg_spectrum[self.divides[1]:])    # 1000+ Гц
            
            if(self.kick.valDiffer>0):
                mids -= self.kick.valDiffer * SIDECHAIN_POWER
                highs -= self.kick.valDiffer * SIDECHAIN_POWER

            bass_fin = int(np.interp(bass, [-100, 0], [0, 255]))
            mids_fin = int(np.interp(mids, [-100, 0], [0, 255]))
            highs_fin = int(np.interp(highs, [-100, 0], [0, 255]))
            
            color_data = str(bass_fin) + ',' + str(mids_fin) + ',' + str(highs_fin) + ';'
                                 
            #ищем точку макс гомкости в установленном диапазоне
            self.kick.pointSearch(self.avg_spectrum)
            self.maxPointsX.append(self.kick.X)            
            self.maxPointsY.append(self.kick.Y)
            #print(self.kick.valDiffer)

            #отправка в сом порт данных
            self.ser.write(color_data.encode()) 
    
    def animProcessor(self):
        try:            
            while True:
                #забираем спектр в локаньную переменную
                self.avg_spectrum = plotter.avg_spectrum                
                #определяем ноты и цвет для них
                notes = self.detector.detect_notes(self.avg_spectrum)
                color = self.detector.get_note_color(notes)
                new_color = list(color)
                for i in range(3):
                    self.color_f[i] += (new_color[i] - self.color_f[i]) * SMOOTH_COEFF
                #фильтрованное += (новое - фильтрованное) * коэффициент:
                color_data = str(int(self.color_f[0])) + ',' + str(int(self.color_f[1])) + ',' + str(int(self.color_f[2])) + ';'
            
                #print(color_data)
                self.ser.write(color_data.encode()) 
                #self.maxPointsX.clear()
                #self.maxPointsY.clear()
                #self.animClassic()
                time.sleep(0.03)
        except Exception as e:
            print(f"Ошибка обновления: {e}")

class PointSidechain:
    def __init__(self, aFreq, bFreq, freq_bins):
        self.aFreq = self.searchGraphIndex(aFreq)
        self.bFreq = self.searchGraphIndex(bFreq)
        self.avg_spectrum = []
        self.freq_bins = freq_bins
        self.X = 0.0
        self.Y = 0.0
        self.minVal = 0.0
        self.lastVal = 0.0
        self.valDiffer = 0.0        
    
    def searchGraphIndex(self, freq):
        for i in range(0, BLOCK_SIZE//2, 1):
           if(freq <= fftAnalyzer.freq_bins[i]):
                return i
        return 0

    def pointMaxSearch(self, indexA, indexB):
        self.maxIndex = -1
        self.maxVal = -150
        for i in range(indexA, indexB, 1):
            if(self.avg_spectrum[i]>self.maxVal):
                self.maxVal = self.avg_spectrum[i]
                self.maxIndex = i
        return self.maxIndex

    def pointSearch(self, avg_spectrum):
        self.avg_spectrum = avg_spectrum
        self.index = self.pointMaxSearch(self.aFreq, self.bFreq)    
        self.lastVal = self.Y    
        self.X = self.freq_bins[self.index]        
        self.Y = self.avg_spectrum[self.index]
        if(self.minVal > self.Y):
            self.minVal = self.Y
        else:
            self.valDiffer= self.Y-self.minVal  

        if(self.Y < self.lastVal):
            self.minVal = self.Y      

def uiInit():
    root = tk.Tk()
    app = AppUI(root)
    root.mainloop()

try:
    print("It try block!") 
    fftAnalyzer = Analyzer(DEVICE_ID, 
                           FS, 
                           BLOCK_SIZE, 
                           CHANNELS,
                           BUFFER_SIZE, 
                           FREQ_RANGE)
    plotter = GraphPlotter(fftAnalyzer.freq_bins, 
                           fftAnalyzer.data_queue)
    animModule = AnimationModule(fftAnalyzer.freq_bins)

    audio_thread = Thread(target=fftAnalyzer.start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start()

    anim_thread = Thread(target=animModule.animProcessor)
    anim_thread.daemon = True
    anim_thread.start()

    #ui_thread = Thread(target = uiInit)
    #ui_thread.daemon = True
    #ui_thread.start()

    # Запуск анимации
    ani = animation.FuncAnimation(
        plotter.fig, 
        plotter.update_plot,
        interval=50,
        blit=True,
        cache_frame_data=False
    )
    plt.show()

except KeyboardInterrupt:
    print("\nОстановка...")
    animModule.ser.close()

finally:
    fftAnalyzer.stop_event.set()
    plt.close()
    print("Программа завершена.")