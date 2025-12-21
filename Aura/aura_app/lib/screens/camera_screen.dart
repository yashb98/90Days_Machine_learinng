import 'dart:typed_data'; // for Uint8List
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

class _CameraScreenState extends State<CameraScreen> with TickerProviderStateMixin, WidgetsBindingObserver {
  // --- STATE VARIABLES ---
  CameraController? controller;
  int _frameCounter = 0;
  bool isStreaming = false;
  String debugStatus = "System Ready";
  String aiStatus = "IDLE";
  String _currentMode = "safety"; 

  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  // --- GEOPOSE FROM ANDROID --- 
  static const _geoChannel = MethodChannel('aura/geospatial');

  Map<String, dynamic> _geospatialPose = {
    'lat': 0.0,
    'lng': 0.0,
    'alt': 0.0,
    'heading': 0.0,
    'hAcc': 0.0,
    'headingAcc': 0.0,
    'source': 'NONE',
  };

  // --- LOCATION VARIABLES ---
  final LocationService _locationService = LocationService();
  String currentLocation = 'Searching...';
  Position? _currentPosition;

  // --- SERVICES ---
  late FlutterTts flutterTts;
  final FirebaseAnalytics _analytics = FirebaseAnalytics.instance;
  final stt.SpeechToText _speech = stt.SpeechToText();
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final PcmAudioService _audioService = PcmAudioService();

  // --- WEBSOCKET CONFIG ---
  // Ensure this IP is correct for your network
  final String _socketUrl = 'ws://192.168.0.61:8080/ws';
  WebSocketChannel? _channel;
  bool _isConnected = false;

  // --- HARDWARE STATE ---
  List<CameraDescription> _cameras = [];
  int _selectedCameraIndex = 0;
  bool _isListening = false;

  // --- ANIMATIONS ---
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this); 
    WidgetsBinding.instance.addObserver(this);
    _audioService.initialize();
    _audioService.start();

    // _setupGeospatialListener();
    // _startLocationTracking();

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
    _requestPermissions(); // Starts the chain: Perms -> Camera -> Location
    _connectWebSocket();
  }

  void _setupGeospatialListener() {
    _geoChannel.setMethodCallHandler((call) async {
      if (call.method == 'onLocationUpdate') {
        final args = call.arguments as Map;
        setState(() {
          _geospatialPose = {
            'lat': (args['lat'] as num).toDouble(),
            'lng': (args['lng'] as num).toDouble(),
            'alt': (args['altitude'] as num?)?.toDouble() ?? 0.0,
            'accuracy': (args['accuracy'] as num).toDouble(),
            'timestamp': args['timestamp'] as int?,
            'source': 'GPS',
          };
        });

        print('📍 Geospatial Update: $_geospatialPose');
      }
    });
  }

  void _startLocationTracking() async {
    try {
      await _geoChannel.invokeMethod('startLocationTracking');
      print('✅ Geospatial tracking started');
    } catch (e) {
      print('❌ Error starting geospatial tracking: $e');
    }
  }


  @override
  void dispose() {
    _pulseController.dispose();
    _channel?.sink.close();
    flutterTts.stop();
    _speech.stop();
    WidgetsBinding.instance.removeObserver(this); 
    controller?.dispose(); 
    controller?.dispose();
    try {
      _geoChannel.invokeMethod('stopLocationTracking');
    } catch (_) {}
    super.dispose();
  }

  // ===========================================================================
  // LIFECYCLE & LOCATION
  // ===========================================================================

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = controller;

    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCameraAtIndex(_selectedCameraIndex); 
    }
  }

  void _getCurrentLocation() async {
    // 1. Call the service
    Position? position = await _locationService.getCurrentLocation();

    // 2. Update UI based on result
    if (mounted) {
      setState(() {
        if (position != null) {
          _currentPosition = position; 
          currentLocation = "Lat: ${position.latitude.toStringAsFixed(4)}, Lng: ${position.longitude.toStringAsFixed(4)}";
          debugStatus = "Location Secured";
        } else {
          currentLocation = "Permission Denied / Error";
          debugStatus = "Loc Error";
        }
      });
    }
  }

  Future<void> _requestPermissions() async {
    // 1. Request Camera & Mic
    Map<Permission, PermissionStatus> statuses = await [
      Permission.camera,
      Permission.microphone,
    ].request();

    if (statuses[Permission.camera]!.isGranted) {
       _initCameras();
    } else {
      setState(() => debugStatus = "Camera Permission Required");
    }

    // 2. Request Location
    _getCurrentLocation();
  }

  // ===========================================================================
  // LOGIC SECTIONS (MODES & WS)
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
      _channel = null; 
      _connectWebSocket();
    }
  }

  Future<void> _connectWebSocket() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      final token = await user?.getIdToken();
      if (token == null) {
          print("❌ Auth Error: No User Token");
          return;
      }

      // Build the URL with Authentication and Mode
      final secureUri = Uri.parse(_socketUrl).replace(queryParameters: {
        'token': token,
        'mode': _currentMode 
      });

      print("🌍 Connecting to: $secureUri");
      _channel = WebSocketChannel.connect(secureUri);
      
      if(mounted) setState(() => _isConnected = true);

      _channel!.stream.listen(
        (message) {
          if (!_isConnected && mounted) setState(() => _isConnected = true);

          // ---------------------------------------------------------
          // CASE A: BINARY AUDIO DATA (PCM 24kHz)
          // ---------------------------------------------------------
          if (message is List<int>) {
             // Pass the raw bytes directly to the audio engine
             // (Ensure _audioService is initialized in initState)
             _audioService.feedAudioChunk(Uint8List.fromList(message));
             return; 
          }

          // ---------------------------------------------------------
          // CASE B: TEXT JSON DATA (Commands & Metadata)
          // ---------------------------------------------------------
          try {
            final data = jsonDecode(message as String);

            // Logging for debugging
            if (data['cmd'] != 'status') { // Reduce noise
                print("🟢 [RECV] ${data['cmd']}");
            }

            if (data['cmd'] == 'speak') {
               String text = data['text'];
               String priority = data['priority'] ?? 'normal';

               // 1. Update UI Text
               if(mounted) {
                   setState(() {
                       debugStatus = text;
                       
                       // 2. Handle Visual Status & Haptics
                       if (priority == 'high' || text.contains("[CRITICAL]")) {
                          aiStatus = "DANGER";
                          HapticFeedback.heavyImpact(); 
                       } else {
                          aiStatus = "SPEAKING";
                       }
                   });
               }

               // ⚠️ CRITICAL: DO NOT CALL _speak(text) HERE.
               // The audio is already playing via Case A (Binary Stream).
               // If you call _speak() here, the local TTS robot will talk over the AI.
            }
            else if (data['cmd'] == 'interrupt') {
               // Stop any lingering audio
               flutterTts.stop(); 
               if(mounted) setState(() => aiStatus = "INTERRUPTED");
            }
            else if (data['cmd'] == 'status') {
               // Backend updating its state (e.g. "THINKING")
               if(mounted) setState(() => aiStatus = data['state'].toString().toUpperCase());
            }

          } catch (e) { 
            print("❌ JSON Parse Error: $e"); 
          }
        }, 
        onError: (e) {
            print("❌ WebSocket Error: $e");
            if(mounted) setState(() => _isConnected = false);
        }, 
        onDone: () {
            print("⚠️ WebSocket Closed by Server");
            if(mounted) setState(() => _isConnected = false);
        }
      );

    } catch (e) { 
        print("❌ Connection Failed: $e");
        if(mounted) setState(() => _isConnected = false);
    }
  }

  // ===========================================================================
  // HARDWARE (CAMERA & AUDIO)
  // ===========================================================================

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

  Future<void> _processFrame(CameraImage image) async {
    if (isProcessingFrame) return; 
    final now = DateTime.now();

    // Throttle: 500ms for safety mode (faster), 1.5s for reading (steadier)
    final throttleDuration = _currentMode == 'reading' 
        ? const Duration(milliseconds: 1500) 
        : const Duration(milliseconds: 500);

    if (lastFrameTime != null && now.difference(lastFrameTime!) < throttleDuration) {
      return; 
    }

    isProcessingFrame = true;
    lastFrameTime = now;

    try {
      _frameCounter++;
      final rawData = {
        'width': image.width,
        'height': image.height,
        'format': image.format.group.name,
        'planes': image.planes.map((plane) => {
          'bytes': plane.bytes,
          'bytesPerRow': plane.bytesPerRow, 
          'bytesPerPixel': plane.bytesPerPixel,
        }).toList(),
      };

      // Run heavy image conversion in background
      final String? base64Result = await compute(convertToBase64Jpeg, rawData);

      if (base64Result != null) {
        if (_channel != null && _isConnected) {
            
            // --- LOGGING ---
            final kbSize = (base64Result.length / 1024).toStringAsFixed(1);
            print("📤 [SEND] Frame #$_frameCounter ($kbSize KB) | Mode: $_currentMode");

            _channel!.sink.add(jsonEncode({
                "image": base64Result,
                "frame_id": _frameCounter,
                "timestamp": DateTime.now().millisecondsSinceEpoch,
                
                // CRITICAL FIX 1: Send "pose" for Backend Memory Service
                "pose": _geospatialPose, 
                
                // CRITICAL FIX 2: Send "intent" to trigger Cloud Vision (OCR)
                "intent": _currentMode == "reading" ? "read_text" : "stream",
                
                // Optional: Fallback GPS if AR is invalid
                "location": _currentPosition != null 
                    ? {"lat": _currentPosition!.latitude, "lng": _currentPosition!.longitude} 
                    : null,
            }));
        }
      }
    } catch (e) { 
        print("❌ Frame Error: $e"); 
    } finally { 
        isProcessingFrame = false; 
    }
  }

  // --- AUDIO ---
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
    if (text.isNotEmpty) {
      if(mounted) setState(() => debugStatus = text);
      await flutterTts.speak(text);

      // Reset status after speaking
      if(mounted) {
         setState(() => aiStatus = isStreaming ? "WATCHING" : "IDLE");
      }
    }
  }

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
          pauseFor: const Duration(milliseconds: 1500),
          listenFor: const Duration(seconds: 30), 
          onResult: (result) {
            if (result.finalResult) {
              String command = result.recognizedWords;

              if (_channel != null) {
                // --- LOGGING SEND (TEXT) ---
                print("🔵 -----------------------------------------------------");
                print("🗣️ [FLUTTER SEND] COMMAND");
                print("   Text: \"$command\"");
                print("-------------------------------------------------------");

                _channel!.sink.add(jsonEncode({ "text": command }));
                setState(() => debugStatus = "You: $command");
              }
              setState(() => _isListening = false);
            }
          },
        );
      }
  }

  // ===========================================================================
  // UI BUILD
  // ===========================================================================
  @override
  Widget build(BuildContext context) {
    // 1. Safety Check (FIXED LOGIC)
    if (controller == null || !controller!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(color: Colors.cyanAccent),
              SizedBox(height: 20),
              Text("Initializing Vision...", style: TextStyle(color: Colors.white54))
            ],
          ),
        ),
      );
    }

    // 2. Calculation for Full Screen Coverage
    final size = MediaQuery.of(context).size;
    var scale = size.aspectRatio * controller!.value.aspectRatio;
    if (scale < 1) scale = 1 / scale;

    return Scaffold(
      key: _scaffoldKey, 
      backgroundColor: Colors.black,
      drawer: MainDrawer(currentMode: _currentMode, onModeSelected: _onModeSelected),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. CAMERA FEED
          Transform.scale(
            scale: scale,
            child: Center(
              child: CameraPreview(controller!),
            ),
          ),

          // 2. SAFETY OVERLAY
          SafetyLayer(aiStatus: aiStatus),

          // 3. TOP STATUS BAR
          StatusBar(
            currentPosition: _currentPosition,
            isConnected: _isConnected,
            onMenuTap: () => _scaffoldKey.currentState?.openDrawer(),
            currentMode: _currentMode,
          ),

          // 4. BOTTOM CONTROL DECK
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