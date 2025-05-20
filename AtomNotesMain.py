# pyright: ignore[reportMissingImports]
import sounddevice as sd
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
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
        self.data_queue = data_queue
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.semilogx(freq_bins, np.zeros_like(freq_bins))
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
                self.line.set_ydata(self.avg_spectrum)
                #animClassic()
        
            return self.line,
        except Exception as e:
            print(f"Ошибка обновления: {e}")
            return self.line,

class AnimationModule:
    def __init__(self):
        print("It init Anim Module!")
        self.divides = []
        #рассчитаем индексы разделительных частот    
        self.divides.append(self.searchGraphIndex(DIVIDE_A))
        self.divides.append(self.searchGraphIndex(DIVIDE_B))

    def searchGraphIndex(self, freq):
        for i in range(0, BLOCK_SIZE//2, 1):
           if(freq <= fftAnalyzer.freq_bins[i]):
                return i
        return 0  
    
    def animClassic(self):
        if(len(plotter.avg_spectrum) > 0):
            bass = np.mean(plotter.avg_spectrum[4:self.divides[0]])    # 20-100 Гц
            mids = np.mean(plotter.avg_spectrum[self.divides[0]:self.divides[1]])  # 100-1000 Гц
            highs = np.mean(plotter.avg_spectrum[self.divides[1]:])    # 1000+ Гц
            bass_fin = int(np.interp(bass, [-100, 0], [0, 255]))
            mids_fin = int(np.interp(mids, [-100, 0], [0, 255]))
            highs_fin = int(np.interp(highs, [-100, 0], [0, 255]))
            
            color_data = str(bass_fin) + ',' + str(mids_fin) + ',' + str(highs_fin) + ';'
            print(color_data)
            #ser.write(color_data.encode()) 
    
    def animProcessor(self):
        try:            
            while True:
                self.animClassic()
                time.sleep(0.03)
        except Exception as e:
            print(f"Ошибка обновления: {e}")

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
    animModule = AnimationModule()

    audio_thread = Thread(target=fftAnalyzer.start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start()

    anim_thread = Thread(target=animModule.animProcessor)
    anim_thread.daemon = True
    anim_thread.start()

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