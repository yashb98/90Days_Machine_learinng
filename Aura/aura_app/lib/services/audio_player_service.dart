
import 'dart:async';
import 'dart:typed_data';
import 'package:sound_stream/sound_stream.dart';

class PcmAudioService {
    final PlayerStream _player = PlayerStream();
    final StreamController<Uint8List> _audioStreamController = StreamController<Uint8List>();
    StreamSubscription<Uint8List>? _audioSubscription;
    bool _isInitialized = false;

    static const int sampleRate = 1600;

    Future<void> initialize() async {
        if(_isInitialized) return;

        // 1. Initialise the player via the plugin
        await _player.initialize(
            sampleRate:sampleRate,
            
        );
        // 2. Connect oue custom stream controller to player's sink
        // This allows us to "feed" chunks into _audioStreamController later

        _audioSubscription = _audioStreamController.stream.listen((chunk) {
            _player.writeChunk(chunk);
        });

        _isInitialized= true;
        print("PCM Audio Engine Initialised at ${sampleRate}Hz");
    } 
    void feedAudioChunk(Uint8List chunk) {
        if(!_isInitialized) {
            print("Warning: Audio player not initialised yet.");
            return;
        }
        // Add the chunk to the stream. The listener above will write it to the hardware.
        _audioStreamController.add(chunk);
    }

    // Call this function whenever you recieve a chunk from the AI Websocket
    Future<void> start() async {
        await _player.start();

    }

    Future<void> stop() async {
        await _player.stop();
    }

    void dispose() {
        _audioSubscription?.cancel();
        _audioStreamController.close();
        _player.dispose();
    }

}
  
