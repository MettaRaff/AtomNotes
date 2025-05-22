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
from threading import Thread, Event

# Конфигурация
DEVICE_ID = 9             # Ваш device_id
FS = 44100                # Частота дискретизации
BLOCK_SIZE = 2048         # Размер блока для БПФ
COM_PORT = 'COM7'         # Замените на ваш порт
BAUDRATE = 115200
CHANNELS = 1              # Количество каналов
BUFFER_SIZE = 1           # Размер буфера для сглаживания
FREQ_RANGE = (20, 20000)  # Диапазон частот
DIVIDE_A = 160.0 #Hz
DIVIDE_B = 2000.0 #Hz

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
        self.y_arr = []
        self.data_queue = data_queue
        self.freq_bins = freq_bins
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.semilogx(self.freq_bins, np.zeros_like(self.freq_bins))
        self.point, = self.ax.plot(100, -30, 'ro', markersize=8)

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
                self.avg_spectrum = np.mean(self.data_queue, axis=0)
                #self.line.set_ydata(self.avg_spectrum)
                #print(animModule.kickPoint)
                
                #self.y_arr.append(animModule.KickPoint[1])
                for i in range(len(self.lines)):
                    if(i == 0):  
                        self.lines[i].set_ydata(self.avg_spectrum)
                    elif(i == 1):
                        if(len(animModule.maxPointsX)>0):
                            self.lines[i].set_data(animModule.maxPointsX,
                                                   animModule.maxPointsY)
                #self.y_arr.clear                    
                #x = self.freq_bins[animModule.KickPoint]
                #y = self.avg_spectrum[animModule.KickPoint]
                #print(f"X: {x} Y: {y}")
                #print(self.freq_bins)
                #self.point.set_data(animModule.KickPoint[0],
                #                    animModule.KickPoint[1])
                #self.point.set_xdata(animModule.KickPoint[0])
                #self.point.set_ydata(animModule.KickPoint[1])
            return self.point, self.line,
        except Exception as e:
            print(f"Ошибка обновления: {e}")
            return self.point, self.line,

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
        print("It init Anim Module!")
        self.divides = []
        self.avg_spectrum = []
        self.freq_bins = freq_bins
        self.maxPointsX = []
        self.maxPointsY = []
        #рассчитаем индексы разделительных частот    
        self.divides.append(self.searchGraphIndex(DIVIDE_A))
        self.divides.append(self.searchGraphIndex(DIVIDE_B))
        self.kickA = self.searchGraphIndex(60)
        self.kickB = self.searchGraphIndex(150)    

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
            bass_fin = int(np.interp(bass, [-100, 0], [0, 255]))
            mids_fin = int(np.interp(mids, [-100, 0], [0, 255]))
            highs_fin = int(np.interp(highs, [-100, 0], [0, 255]))
            
            color_data = str(bass_fin) + ',' + str(mids_fin) + ',' + str(highs_fin) + ';'
            
            self.kickSearch()   #ищем точку макс гомкости в басах                       
            
            #print(color_data)
            #ser.write(color_data.encode()) 

    def pointMaxSearch(self, indexA, indexB):
        self.maxIndex = -1
        self.maxVal = -150
        for i in range(indexA, indexB, 1):
            if(self.avg_spectrum[i]>self.maxVal):
                self.maxVal = self.avg_spectrum[i]
                self.maxIndex = i
        return self.maxIndex
    
    def kickSearch(self):
        self.index = self.pointMaxSearch(self.kickA, self.kickB)
        self.maxPointsX.append(self.freq_bins[self.index])            
        self.maxPointsY.append(self.avg_spectrum[self.index]) 
    
    def animProcessor(self):
        try:            
            while True:
                self.avg_spectrum = plotter.avg_spectrum
                self.maxPointsX.clear()
                self.maxPointsY.clear()
                self.animClassic()
                time.sleep(0.03)
        except Exception as e:
            print(f"Ошибка обновления: {e}")

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

finally:
    fftAnalyzer.stop_event.set()
    plt.close()
    print("Программа завершена.")