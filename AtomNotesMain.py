# pyright: ignore[reportMissingImports]
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from threading import Thread, Event

# Конфигурация
DEVICE_ID = 1          # Ваш device_id
FS = 44100             # Частота дискретизации
BLOCK_SIZE = 2048      # Размер блока для БПФ
CHANNELS = 1           # Количество каналов
BUFFER_SIZE = 3       # Размер буфера для сглаживания
FREQ_RANGE = (20, 20000)  # Диапазон частот

# Инициализация данных
freq_bins = np.fft.fftfreq(BLOCK_SIZE, 1/FS)[:BLOCK_SIZE//2]
data_queue = deque(maxlen=BUFFER_SIZE)
stop_event = Event()

# Настройка графика
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.semilogx(freq_bins, np.zeros_like(freq_bins))
ax.set_xlim(FREQ_RANGE)
ax.set_ylim(-80, 0)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (dB)')
ax.set_title('Real-Time Audio Spectrum')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
fig.tight_layout()

def audio_callback(indata, frames, time, status):
    """Обработка аудио данных"""
    if status:
        print(f"Audio error: {status}")
    
    # Преобразование в моно
    #signal = indata[:,0] if indata.ndim > 1 else indata.flatten()
    # Конвертируем в моно, если нужно
    if indata.ndim > 1:
        signal = np.mean(indata, axis=1)
    else:
        signal = indata.flatten()
    signal = half_array(signal)
    # Оконная функция Ханна
    window = np.hanning(len(signal))
    windowed = signal * window
    # БПФ и преобразование в дБ
    fft = np.fft.fft(windowed)    
    magnitude = np.abs(fft) * 2 / np.sum(window)
    db_spectrum = 20 * np.log10(magnitude + 1e-12)
    data_queue.append(db_spectrum)    
    #data_queue = db_spectrum

def half_array(arr_inp):
    arr_out = []
    if len(arr_inp) <= 1:
        return arr_inp
    for i in range(0, len(arr_inp)-1, 2):
        half_val = (arr_inp[i] + arr_inp[i+1]) / 2
        arr_out.append(half_val)
    return arr_out

def update_plot(frame):
    #global data_queue
    """Обновление графика"""
    try:
        if data_queue:
            #print("try:")
            # Усреднение по буферу
            avg_spectrum = np.mean(data_queue, axis=0)
            #print(f"x leng: {len(freq_bins)}")
            #print(f"y leng: {len(avg_spectrum)}")
            line.set_ydata(avg_spectrum)
            
            
            # Динамическое обновление цвета
            #avg_power = np.mean(avg_spectrum)
            #color = plt.cm.viridis((avg_power + 80) / 80)
            #line.set_color(color)
            
        return line,
    except Exception as e:
        print(f"Ошибка обновления: {e}")
        return line,

def start_audio_stream():
    """Запуск аудио потока"""
    with sd.InputStream(
        device=DEVICE_ID,
        samplerate=FS,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        callback=audio_callback
    ):
        print("Аудиопоток запущен...")
        while not stop_event.is_set():
            sd.sleep(10)

try:
    # Запуск аудио потока
    audio_thread = Thread(target=start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start()
    # Запуск анимации
    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        interval=50,
        blit=True,
        cache_frame_data=False
    )
    plt.show()

except KeyboardInterrupt:
    print("\nОстановка...")
finally:
    stop_event.set()
    plt.close()
    print("Программа завершена.")