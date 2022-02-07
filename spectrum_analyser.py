import numpy as np 
import pyaudio as pa 
import struct 
import matplotlib.pyplot as plt 


FORMAT = pa.paInt16
CHANNELS = 1
RATE = 48000 # in Hz
INTERVAL = 0.05
CHUNK = int(RATE * INTERVAL)

MAX_WINDOW_SIZE = 10

p = pa.PyAudio()

stream = p.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

fig, (ax,ax1) = plt.subplots(2)
x_fft = np.linspace(0, RATE, CHUNK)
x = np.arange(0,2*CHUNK,2)
line, = ax.plot(x, np.random.rand(CHUNK),'r')
line_fft, = ax1.semilogx(x_fft, np.random.rand(CHUNK), 'b')
line_fft_peak, = ax1.semilogx(x_fft, np.random.rand(CHUNK), 'r')
ax.set_ylim(-32000,32000)
ax.ser_xlim = (0,CHUNK)
ax1.set_xlim(20,RATE/2)
ax1.set_ylim(0,.02)
fig.show()


fft_linedata_history = []
first_run = True


while True:
    data = stream.read(CHUNK)
    dataInt = struct.unpack(str(CHUNK) + 'h', data)
    line.set_ydata(dataInt)

    # fft_linedata = np.abs(np.fft.fft(dataInt))*2/(11000*CHUNK)
    fft_linedata = np.abs(np.fft.fft(dataInt))*2/(44000*CHUNK)

    if(first_run):
        fft_linedata_history = np.array([fft_linedata, fft_linedata])
        # print(fft_linedata_history)
        first_run = False
    
    fft_linedata_history = np.append(fft_linedata_history, [fft_linedata], axis=0)

    if len(fft_linedata_history) >= MAX_WINDOW_SIZE:
        fft_linedata_history = np.delete(fft_linedata_history, 0,0)

    peak_fft =np.zeros(len(fft_linedata_history[0]), dtype=np.float64)
    for i in range(len(fft_linedata_history[0])):
        peak_fft[i] = max(np.take(fft_linedata_history, i, axis=1))

    line_fft.set_ydata(fft_linedata)
    line_fft_peak.set_ydata(peak_fft)
    fig.canvas.draw()
    fig.canvas.flush_events()

stream.stop_stream()
stream.close()
p.terminate()