import threading
import queue
import time


class AudioWorker:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_running = True

        # Start the background thread immediately
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def _play_loop(self):
        """Internal loop that runs in the background."""
        while self.is_running:
            try:
                # 1. Get the next audio task
                audio_task = self.audio_queue.get(timeout=0.1)

                # 2. Check for interruption BEFORE starting
                if self.stop_event.is_set():
                    self.audio_queue.task_done()
                    continue

                # 3. Simulate Playing (Replace this with actual PyAudio/TTS later)
                # We break the audio into small chunks so we can interrupt MID-SENTENCE
                print(f"Speaking: {audio_task}...")
                for _ in range(10):
                    if self.stop_event.is_set():
                        print("Audio CUT OFF!")
                        break
                    time.sleep(0.1)  # Simulate length of audio chunk

                self.audio_queue.task_done()

            except queue.Empty:
                continue

    def speak(self, text_or_audio):
        """Add speech to the queue."""
        self.stop_event.clear()  # Ensure channel is open
        self.audio_queue.put(text_or_audio)

    def stop(self):
        """The 'Barge-In' trigger."""
        self.stop_event.set()
        # Optional: Clear the queue so it doesn't say old stuff later
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
