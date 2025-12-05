
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui'; // Required for Glassmorphism
import 'package:flutter/foundation.dart'; // For compute()
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:flutter/services.dart'; // For Haptics
import 'package:geolocator/geolocator.dart'; 
import '../services/location_service.dart'; 

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

// Added TickerProviderStateMixin for Animations
class _CameraScreenState extends State<CameraScreen> with TickerProviderStateMixin {
  CameraController? controller;
  bool isStreaming = false;
  String debugStatus = "System Ready"; // Subtitles
  String aiStatus = "IDLE"; // Controls colors (IDLE, THINKING, SPEAKING, DANGER)
  
  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  // Services
  late FlutterTts flutterTts;
  final FirebaseAnalytics _analytics = FirebaseAnalytics.instance;
  final LocationService _locationService = LocationService();
  
  // WebSocket
  final String _socketUrl = 'ws://192.168.0.61:8081/ws';
  WebSocketChannel? _channel;
  bool _isConnected = false;

  // State
  List<CameraDescription> _cameras = [];
  int _selectedCameraIndex = 0;
  Position? _currentPosition;
  StreamSubscription<Position>? _positionStreamSubscription;

  // Animation Controllers (NEW)
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    
    // --- ANIMATION SETUP ---
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _initTts();
    _requestPermissions();
    _connectWebSocket();
    _startLocationUpdates();
  }

  // --- LOGIC SECTION (Your working code) ---

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

  Future<void> _connectWebSocket() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      final token = await user?.getIdToken();
      if (token == null) return;

      final secureUri = Uri.parse(_socketUrl).replace(queryParameters: {'token': token});
      print("🔌 Connecting to: $secureUri");

      _channel = WebSocketChannel.connect(secureUri);
      if(mounted) setState(() => _isConnected = true);

      _channel!.stream.listen((message) {
        if (!_isConnected && mounted) setState(() => _isConnected = true);

        try {
          final data = jsonDecode(message);
          
          if (data['cmd'] == 'speak') {
             String text = data['text'];
             print("🗣️ AI: $text");
             
             // URGENCY HANDLING
             if (text.contains("[CRITICAL]")) {
                HapticFeedback.heavyImpact(); 
                HapticFeedback.heavyImpact();
                text = text.replaceFirst("[CRITICAL]", "Warning! ");
                if(mounted) setState(() => aiStatus = "DANGER");
             } else if(mounted) {
                setState(() => aiStatus = "SPEAKING");
             }

             // Update Subtitles
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
      }, onError: (e) {
        if(mounted) setState(() => _isConnected = false);
      }, onDone: () {
        if(mounted) setState(() => _isConnected = false);
      });
    } catch (e) { print(e); }
  }
  
  Future<void> _speak(String text) async {
    await flutterTts.stop(); 
    if (text.isNotEmpty) await flutterTts.speak(text);
  }

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
      // Reconnect socket to clear buffers on switch
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

  Future<void> _processFrame(CameraImage image) async {
    if (isProcessingFrame) return; 
    final now = DateTime.now();
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

  @override
  void dispose() {
    _pulseController.dispose(); // Dispose animation
    _positionStreamSubscription?.cancel();
    _channel?.sink.close();
    flutterTts.stop();
    controller?.dispose();
    super.dispose();
  }

  // --- UI BUILDER (Senior Frontend Design) ---
  @override
  Widget build(BuildContext context) {
    Color statusColor = _getStatusColor(aiStatus);

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. CAMERA FEED 
          if (controller != null && controller!.value.isInitialized)
            Container(
              color: Colors.black, // Background for letterboxing
              child: Center(
                child: CameraPreview(controller!),
              ),
            )
          else
            const Center(child: CircularProgressIndicator(color: Colors.cyanAccent)),

          // 2. SAFETY OVERLAY (Flashes Red on Danger)
          IgnorePointer(
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              decoration: BoxDecoration(
                border: Border.all(
                  color: aiStatus == "DANGER" ? Colors.red.withOpacity(0.6) : Colors.transparent,
                  width: 12,
                ),
              ),
            ),
          ),

          // 3. TOP GLASS BAR (Status & GPS)
          Positioned(
            top: 50, left: 20, right: 20,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  color: Colors.black.withOpacity(0.4),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(
                            _currentPosition != null ? Icons.location_on : Icons.location_searching,
                            color: _currentPosition != null ? Colors.greenAccent : Colors.grey,
                            size: 16,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            _currentPosition != null ? "GPS Active" : "Locating...",
                            style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: _isConnected ? Colors.green.withOpacity(0.2) : Colors.red.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: _isConnected ? Colors.green : Colors.red),
                        ),
                        child: Text(
                          _isConnected ? "ONLINE" : "OFFLINE",
                          style: TextStyle(
                            color: _isConnected ? Colors.greenAccent : Colors.redAccent, 
                            fontSize: 10, fontWeight: FontWeight.bold
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          // 4. BOTTOM CONTROL DECK (Glassmorphism)
          Positioned(
            bottom: 30, left: 20, right: 20,
            child: Column(
              children: [
                // AI STATUS ORB (Animated)
                AnimatedBuilder(
                  animation: _pulseController,
                  builder: (context, child) {
                    return Container(
                      height: 80, width: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: statusColor.withOpacity(0.1),
                        border: Border.all(color: statusColor.withOpacity(0.5), width: 2),
                        boxShadow: [
                          BoxShadow(
                            color: statusColor.withOpacity(0.3 * _pulseAnimation.value),
                            blurRadius: 20 * _pulseAnimation.value,
                            spreadRadius: 5 * _pulseAnimation.value,
                          )
                        ],
                      ),
                      child: Icon(
                        aiStatus == "DANGER" ? Icons.warning : 
                        aiStatus == "SPEAKING" ? Icons.graphic_eq : 
                        aiStatus == "WATCHING" ? Icons.remove_red_eye : Icons.circle,
                        color: statusColor,
                        size: 32,
                      ),
                    );
                  },
                ),
                
                const SizedBox(height: 20),

                // SUBTITLES (Smooth Transition)
                AnimatedSwitcher(
                  duration: const Duration(milliseconds: 300),
                  child: Container(
                    key: ValueKey<String>(debugStatus),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.6),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      debugStatus, // Shows what AI is saying
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white, fontSize: 14),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // CONTROLS
                ClipRRect(
                  borderRadius: BorderRadius.circular(30),
                  child: BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 20),
                      color: Colors.white.withOpacity(0.1),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          IconButton(
                            icon: const Icon(Icons.switch_camera_rounded, color: Colors.white),
                            onPressed: _switchCamera,
                          ),
                          FloatingActionButton(
                            backgroundColor: isStreaming ? Colors.redAccent : Colors.greenAccent,
                            onPressed: _toggleStream,
                            child: Icon(isStreaming ? Icons.stop : Icons.play_arrow, color: Colors.black),
                          ),
                          IconButton(
                            icon: const Icon(Icons.volume_up_rounded, color: Colors.white),
                            onPressed: () => flutterTts.speak("Audio Check"),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getStatusColor(String status) {
    switch (status) {
      case "DANGER": return Colors.redAccent;
      case "SPEAKING": return Colors.cyanAccent;
      case "THINKING": return Colors.amber;
      case "INTERRUPTED": return Colors.orange;
      case "WATCHING": return Colors.purpleAccent;
      default: return Colors.grey;
    }
  }
}

// --- ISOLATE FUNCTION (Unchanged) ---
Future<String?> convertToBase64Jpeg(Map<String, dynamic> data) async {
  try {
    final width = data['width'] as int;
    final height = data['height'] as int;
    final format = data['format'] as ImageFormatGroup;
    final planes = data['planes'] as List;

    img.Image? image;

    if (format == ImageFormatGroup.yuv420) {
      image = img.Image(width: width, height: height, numChannels: 1); 
      final yPlane = planes[0]['bytes'] as Uint8List;
      final int bytesPerRow = planes[0]['bytesPerRow'] as int; 

      for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
          final int yIndex = y * bytesPerRow + x;
          if (yIndex < yPlane.length) {
             final pixel = yPlane[yIndex];
             image.setPixelRgb(x, y, pixel, pixel, pixel);
          }
        }
      }
    } else if (format == ImageFormatGroup.bgra8888) {
      final bytes = planes[0]['bytes'] as Uint8List;
      image = img.Image.fromBytes(
        width: width, 
        height: height, 
        bytes: bytes.buffer,
        order: img.ChannelOrder.bgra
      );
    }

    if (image == null) return null;
    final resized = img.copyResize(image, width: 640); 
    return base64Encode(img.encodeJpg(resized, quality: 70));
  } catch (e) { return null; }
}