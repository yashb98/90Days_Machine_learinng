import 'dart:async';
import 'dart:convert';
import 'dart:ui'; // For ImageFilter
import 'package:flutter/foundation.dart'; // For compute
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // For Haptics
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:geolocator/geolocator.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

// --- CUSTOM IMPORTS ---
import '../services/audio_player_service.dart';
import '../services/location_service.dart';
import '../utils/image_converter.dart';
import '../widgets/camera_feed.dart';
import '../widgets/control_deck.dart';
import '../widgets/status_bar.dart';
import '../widgets/main_drawer.dart';
import '../widgets/safety_layer.dart';


class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with TickerProviderStateMixin {
  // --- STATE VARIABLES ---
  CameraController? controller;
  bool isStreaming = false;
  String debugStatus = "System Ready";
  String aiStatus = "IDLE";
  String _currentMode = "safety"; 
  
  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  // --- SERVICES ---
  late FlutterTts flutterTts;
  final FirebaseAnalytics _analytics = FirebaseAnalytics.instance;
  final LocationService _locationService = LocationService();
  final stt.SpeechToText _speech = stt.SpeechToText();
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  
  // --- WEBSOCKET CONFIG ---
  // REPLACE WITH YOUR ACTUAL CLOUD RUN URL
  final String _socketUrl = 'ws://192.168.0.61:8080/ws';
  // final String _socketUrl = 'wss://aura-backend-service-963226949438.europe-west2.run.app/ws';
  WebSocketChannel? _channel;
  bool _isConnected = false;

  // --- HARDWARE STATE ---
  List<CameraDescription> _cameras = [];
  int _selectedCameraIndex = 0;
  Position? _currentPosition;
  StreamSubscription<Position>? _positionStreamSubscription;
  bool _isListening = false;

  // --- ANIMATIONS ---
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    
    // 1. Setup Animations
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    // 2. Initialize Features
    _initTts();
    _initSpeech();
    _requestPermissions();
    _connectWebSocket();
    _startLocationUpdates();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _positionStreamSubscription?.cancel();
    _channel?.sink.close();
    flutterTts.stop();
    _speech.stop();
    controller?.dispose();
    super.dispose();
  }

  // ===========================================================================
  // LOGIC SECTIONS
  // ===========================================================================
  void _onModeSelected(String mode) {
    if (_currentMode != mode) {
      setState(() {
        _currentMode = mode;
        debugStatus = "Switched to ${mode.toUpperCase()}";
        aiStatus = "IDLE";
      });
      // Reconnect to backend with new mode
      _channel?.sink.close();
      _connectWebSocket();
    }
  }

  void _showModeMenu() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.9),
          borderRadius: const BorderRadius.vertical(top: Radius.circular(25)),
          border: Border(top: BorderSide(color: Colors.white.withOpacity(0.2), width: 1)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text("SELECT VISION MODE", style: TextStyle(color: Colors.white54, fontSize: 12, fontWeight: FontWeight.bold)),
            const SizedBox(height: 20),
            
            _buildModeTile(Icons.shield_outlined, "Safety Guide", "safety", Colors.greenAccent),
            _buildModeTile(Icons.menu_book_rounded, "Text Reader", "reading", Colors.blueAccent),
            _buildModeTile(Icons.landscape_rounded, "Scenery Description", "scenery", Colors.purpleAccent),
          ],
        ),
      ),
    );
    
  }

  Widget _buildModeTile(IconData icon, String title, String modeKey, Color color) {
    bool isSelected = _currentMode == modeKey;
    return ListTile(
      leading: Icon(icon, color: isSelected ? color : Colors.white54, size: 30),
      title: Text(title, style: TextStyle(color: isSelected ? Colors.white : Colors.white60, fontSize: 18, fontWeight: FontWeight.bold)),
      trailing: isSelected ? Icon(Icons.check_circle, color: color) : null,
      onTap: () {
        Navigator.pop(context); // Close menu
        if (!isSelected) {
          setState(() {
            _currentMode = modeKey;
            debugStatus = "Switching to $title...";
            aiStatus = "IDLE";
          });
          // Reconnect with new mode
          _channel?.sink.close();
          _connectWebSocket();
        }
      },
    );
  }
  // --- 1. LOCATION ---
  void _startLocationUpdates() {
    try {
      _positionStreamSubscription = _locationService.getPositionStream().listen(
        (Position position) {
          if (mounted) setState(() => _currentPosition = position);
        },
        onError: (e) => print("Location Error: $e"),
      );
    } catch (e) {
      print("Error starting location stream: $e");
    }
  }

  // --- 2. TEXT-TO-SPEECH (VOICE) ---
  void _initTts() async {
    flutterTts = FlutterTts();
    await flutterTts.setLanguage("en-US");
    await flutterTts.setPitch(1.0);
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.awaitSpeakCompletion(true);
    await flutterTts.setIosAudioCategory(IosTextToSpeechAudioCategory.playback, [
      IosTextToSpeechAudioCategoryOptions.defaultToSpeaker,
      IosTextToSpeechAudioCategoryOptions.duckOthers
    ]);
  }
  
  Future<void> _speak(String text) async {
    await flutterTts.stop(); 
    if (text.isNotEmpty) await flutterTts.speak(text);
  }

  // --- 3. SPEECH RECOGNITION (EARS) ---
  void _initSpeech() async {
    try {
      await _speech.initialize(
        onStatus: (status) {
          if (status == 'done' || status == 'notListening') {
            setState(() => _isListening = false);
          }
        },
        onError: (e) => print('Speech Error: $e'),
      );
    } catch (e) {
      print("Speech Init Failed: $e");
    }
  }

  void _toggleListening() async {
      HapticFeedback.mediumImpact();

      if (_isListening) {
        _speech.stop();
        setState(() => _isListening = false);
      } else {
        if (!_isConnected) {
          _speak("System offline.");
          return;
        }
        
        if (!isStreaming) _toggleStream(); 
        
        setState(() => _isListening = true);
        flutterTts.stop(); 
        
        _speech.listen(
          // --- FIX 2: Wait longer for pauses ---
          pauseFor: const Duration(seconds: 5), // Wait 5s before stopping
          listenFor: const Duration(seconds: 30), // Max 30s command
          // -------------------------------------
          onResult: (result) {
            if (result.finalResult) {
              String command = result.recognizedWords;
              print("🗣️ Heard: $command");
              
              if (_channel != null) {
                _channel!.sink.add(jsonEncode({ "text": command }));
                setState(() => debugStatus = "You: $command");
              }
              
              // Note: We DON'T set _isListening = false here automatically 
              // if you want to keep the mic open, but for now standard behavior 
              // is to close it after a final result.
              setState(() => _isListening = false);
            }
          },
        );
      }
    }
  // --- 4. WEBSOCKET CONNECTION ---
Future<void> _connectWebSocket() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      final token = await user?.getIdToken();
      if (token == null) return;

      final secureUri = Uri.parse(_socketUrl).replace(queryParameters: {
        'token': token,
        'mode': _currentMode 
      });
      
      _channel = WebSocketChannel.connect(secureUri);
      if(mounted) setState(() => _isConnected = true);

      _channel!.stream.listen((message) {
        if (!_isConnected && mounted) setState(() => _isConnected = true);

        // --- FIX 1: Don't let AI interrupt the USER ---
        if (_isListening) {
          print("🤫 AI stayed silent because User is speaking");
          return; 
        }
        // ----------------------------------------------

        try {
          final data = jsonDecode(message);
          
          if (data['cmd'] == 'speak') {
             String text = data['text'];
             String priority = data['priority'] ?? 'normal';
             
             if (priority == 'high' || text.contains("[CRITICAL]")) {
                HapticFeedback.heavyImpact(); 
                text = text.replaceFirst("[CRITICAL]", "Warning! ");
                if(mounted) setState(() => aiStatus = "DANGER");
             } else if(mounted) {
                setState(() => aiStatus = "SPEAKING");
             }
             
             if(mounted) setState(() => debugStatus = text);
             _speak(text);
          }
          else if (data['cmd'] == 'interrupt') {
            flutterTts.stop();
            if(mounted) setState(() => aiStatus = "INTERRUPTED");
          }
          else if (data['cmd'] == 'status') {
             if(mounted) setState(() => aiStatus = data['state'].toString().toUpperCase());
          }
    } catch (e) { print(e); }
          }, onError: (e) => print(e), onDone: () => print("Closed"));
        } catch (e) { print(e); }
      }
  // --- 5. MODE TOGGLE ---
  void _toggleMode() {
    setState(() {
      _currentMode = (_currentMode == "safety") ? "scenery" : "safety";
      debugStatus = "Switching to ${_currentMode.toUpperCase()}...";
      aiStatus = "IDLE";
    });
    // Reconnect to send new mode to backend
    _channel?.sink.close();
    _connectWebSocket();
  }

  // --- 6. CAMERA CONTROL ---
  Future<void> _requestPermissions() async {
    await [Permission.camera, Permission.microphone, Permission.locationWhenInUse].request();
    _initCameras();
  }

  Future<void> _initCameras() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isNotEmpty) _initializeCameraAtIndex(0);
    } catch (e) { print(e); }
  }

  Future<void> _initializeCameraAtIndex(int index) async {
    if (_cameras.isEmpty) return;
    controller = CameraController(
      _cameras[index],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller!.initialize();
    if (!mounted) return;
    setState(() => _selectedCameraIndex = index);
  }

  void _switchCamera() async {
    if (_cameras.length < 2) return;
    int newIndex = (_selectedCameraIndex + 1) % _cameras.length;
    bool wasStreaming = isStreaming;
    if (isStreaming) {
      await controller?.stopImageStream();
      setState(() => isStreaming = false);
    }
    await controller?.dispose();
    await _initializeCameraAtIndex(newIndex);
    if (wasStreaming) {
      _channel?.sink.close();
      await _connectWebSocket();
      _toggleStream();
    }
  }

  void _toggleStream() {
    if (controller == null || !controller!.value.isInitialized) return;
    if (isStreaming) {
      controller!.stopImageStream();
      setState(() {
        isStreaming = false;
        aiStatus = "IDLE";
      });
    } else {
      controller!.startImageStream((CameraImage image) {
        _processFrame(image);
      });
      setState(() {
        isStreaming = true;
        aiStatus = "WATCHING";
      });
    }
  }

  // --- 7. FRAME PROCESSING ---
  Future<void> _processFrame(CameraImage image) async {
    if (isProcessingFrame) return; 
    final now = DateTime.now();
    
    // Throttle: 1.5 seconds
    if (lastFrameTime != null && 
        now.difference(lastFrameTime!) < const Duration(milliseconds: 1500)) {
      return; 
    }

    isProcessingFrame = true;
    lastFrameTime = now;

    try {
      final rawData = {
        'width': image.width,
        'height': image.height,
        'format': image.format.group,
        'planes': image.planes.map((plane) => {
          'bytes': plane.bytes,
          'bytesPerRow': plane.bytesPerRow, 
          'bytesPerPixel': plane.bytesPerPixel,
        }).toList(),
      };

      // Run heavy compression in background thread using Utility
      final String? base64Result = await compute(convertToBase64Jpeg, rawData);

      if (base64Result != null) {
        if (_channel != null && _isConnected) {
            Map<String, double>? locationData;
            if (_currentPosition != null) {
              locationData = {'lat': _currentPosition!.latitude, 'lng': _currentPosition!.longitude};
            }
            
            _channel!.sink.add(jsonEncode({
                "image": base64Result,
                "timestamp": DateTime.now().millisecondsSinceEpoch,
                "location": locationData 
            }));
        }
      }
    } catch (e) { print(e); } 
    finally { isProcessingFrame = false; }
  }

  // ===========================================================================
  // UI CONSTRUCTION
  // ===========================================================================
@override
  Widget build(BuildContext context) {

    return Scaffold(
      key: _scaffoldKey, // Essential for Drawer
      backgroundColor: Colors.black,
      drawer: MainDrawer(currentMode: _currentMode, onModeSelected: _onModeSelected),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. CAMERA FEED (FIXED: Wrapped in Center to prevent stretching)
          // This ensures the image isn't distorted, though you might see black bars.
          if (controller != null && controller!.value.isInitialized)
            Center(
              child: CameraPreview(controller!),
            )
          else
            const Center(child: CircularProgressIndicator(color: Colors.cyanAccent)),

          // 2. SAFETY OVERLAY (Flashes Red on Danger)
          SafetyLayer(aiStatus: aiStatus),

          // 3. TOP STATUS BAR (Glass UI)
          StatusBar(
            currentPosition: _currentPosition,
            isConnected: _isConnected,
            onMenuTap: () => _scaffoldKey.currentState?.openDrawer(),
            currentMode: _currentMode,
          ),

          // 4. BOTTOM CONTROL DECK (Buttons & Orb)
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) {
              return ControlDeck(
                statusText: debugStatus,
                aiStatus: aiStatus,
                isStreaming: isStreaming,
                isListening: _isListening,
                pulseAnimation: _pulseAnimation,
                onSwitchCamera: _switchCamera,
                onToggleStream: _toggleStream,
                onToggleMic: _toggleListening,
              );
            },
          ),
        ],
      ),
    );
  }
}