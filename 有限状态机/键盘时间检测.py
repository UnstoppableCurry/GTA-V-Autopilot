from pynput import keyboard
import time


def on_press(key):
    global last_time
    last_time = time.time()
    return False


def on_release(key):
    print('release', key, time.time() - last_time)
    return False


while 1:
    with keyboard.Listener(
            on_press=on_press) as listener:
        listener.join()
    with keyboard.Listener(
            on_release=on_release) as listener:
        listener.join()
