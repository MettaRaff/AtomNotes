# pyright: ignore[reportMissingImports]
import sounddevice as sd
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from threading import Thread, Event

# Конфигурация
DEVICE_ID = 1             # Ваш device_id
FS = 44100                # Частота дискретизации
BLOCK_SIZE = 2048         # Размер блока для БПФ
COM_PORT = 'COM7'         # Замените на ваш порт
BAUDRATE = 115200
CHANNELS = 1              # Количество каналов
BUFFER_SIZE = 1           # Размер буфера для сглаживания
FREQ_RANGE = (20, 20000)  # Диапазон частот
DIVIDE_A = 160.0 #Hz
DIVIDE_B = 2000.0 #Hz

# Инициализация UART
ser = serial.Serial(COM_PORT, BAUDRATE, timeout=1)

# Инициализация данных
freq_bins = np.fft.fftfreq(BLOCK_SIZE, 1/FS)[:BLOCK_SIZE//2]
data_queue = deque(maxlen=BUFFER_SIZE)
stop_event = Event()
avg_spectrum = []
divides = []

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

def searchGraphIndex(freq):
    for i in range(0, BLOCK_SIZE//2, 1):
        if(freq <= freq_bins[i]):
            return i
    return 0    
    
#рассчитаем индексы разделительных частот    
divides.append(searchGraphIndex(DIVIDE_A))
divides.append(searchGraphIndex(DIVIDE_B))

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


def map_value(value, in_min, in_max, out_min, out_max, clamp=False):
    # Вычисляем пропорцию
    in_range = in_max - in_min
    out_range = out_max - out_min

    if in_range == 0:
        return out_min  # или raise ValueError("Диапазон ввода не может быть нулевым")

    scaled = (value - in_min) / in_range
    if clamp:
        scaled = max(0.0, min(1.0, scaled))  # Ограничение в пределах [0, 1]

    return out_min + (scaled * out_range)

def animClassic():
    print("try")
    bass = np.mean(avg_spectrum[0:divides[0]])    # 20-100 Гц
    mids = np.mean(avg_spectrum[divides[0]:divides[1]])  # 100-1000 Гц
    highs = np.mean(avg_spectrum[divides[1]:])    # 1000+ Гц

    #bass = map_value(bass, -80, 0, 0, 255)

    #bass = np.interp(bass, [-120, 0], [0, 255])

    bass = abs(bass * 2)
    mids = abs(mids * 2)
    highs = abs(highs * 2)
    
    print("bass : {bass}  mids : {mids}  highs : {highs}") 

def update_plot(frame):
    global avg_spectrum
    """Обновление графика"""
    try:
        if data_queue:
            avg_spectrum = np.mean(data_queue, axis=0)
            line.set_ydata(avg_spectrum)
            
            # Динамическое обновление цвета
            #avg_power = np.mean(avg_spectrum)
            #color = plt.cm.viridis((avg_power + 80) / 80)
            #line.set_color(color)
        bass = np.mean(avg_spectrum[4:divides[0]])    # 20-100 Гц
        mids = np.mean(avg_spectrum[divides[0]:divides[1]])  # 100-1000 Гц
        highs = np.mean(avg_spectrum[divides[1]:])    # 1000+ Гц
        bass_fin = int(np.interp(bass, [-100, 0], [0, 255]))
        mids_fin = int(np.interp(mids, [-100, 0], [0, 255]))
        highs_fin = int(np.interp(highs, [-100, 0], [0, 255]))
        #print("bass :", {bass_fin},  "mids :", {mids_fin}, " highs :", {highs_fin}) 
        color_data = str(bass_fin) + ',' + str(mids_fin) + ',' + str(highs_fin) + ';'
        #print(color_data)
        ser.write(color_data.encode())    
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
    animClassic

except KeyboardInterrupt:
    print("\nОстановка...")
    ser.close()

finally:
    stop_event.set()
    plt.close()
    print("Программа завершена.")