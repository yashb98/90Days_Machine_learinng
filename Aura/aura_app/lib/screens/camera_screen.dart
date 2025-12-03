import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
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
import 'package:geolocator/geolocator.dart'; // for geolocation
import '../services/location_service.dart'; // for location service function

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? controller;
  bool isStreaming = false;
  String debugStatus = "Initializing...";
  String aiStatus = "IDLE";
  
  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  // TTS & Analytics
  late FlutterTts flutterTts;
  final FirebaseAnalytics _analytics = FirebaseAnalytics.instance;
  
  // WebSocket
  final String _socketUrl = 'wss://aura-backend-service-963226949438.europe-west2.run.app/ws';
  WebSocketChannel? _channel;
  bool _isConnected = false;

  // Location Variables  
  final LocationService _locationService = LocationService();
  Position? _currentPosition;
  StreamSubscription<Position>? _positionStreamSubscription;

  // CAMERA SWITCHING STATE
  List<CameraDescription> _cameras = [];
  int _selectedCameraIndex = 0;

  @override
  void initState() {
    super.initState();
    _initTts();
    _requestPermissions();
    _connectWebSocket();
    _startLocationUpdates(); // Start the stream 
  }

  void _initTts() async {
    flutterTts = FlutterTts();
    await flutterTts.setLanguage("en-US");
    await flutterTts.setPitch(1.0);
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.awaitSpeakCompletion(true);

    // Ensure audio plays on media stream (Loud)
    await flutterTts.setIosAudioCategory(IosTextToSpeechAudioCategory.playback, [
      IosTextToSpeechAudioCategoryOptions.defaultToSpeaker,
      IosTextToSpeechAudioCategoryOptions.duckOthers
    ]);
    
    flutterTts.setStartHandler(() => print("TTS Started"));
    flutterTts.setCompletionHandler(() => print("TTS Finished"));
    flutterTts.setErrorHandler((msg) => print("TTS Error: $msg"));
  }

  Future<void> _connectWebSocket() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      final token = await user?.getIdToken();

      if (token == null) {
        print("No Auth Token Found. Cannot connect.");
        return;
      }

      final secureUri = Uri.parse(_socketUrl).replace(queryParameters: {'token': token});
      print("🔌 Connecting to: $secureUri");

      _channel = WebSocketChannel.connect(secureUri);
      if(mounted) setState(() => _isConnected = true);

      _channel!.stream.listen((message) {
        if (!_isConnected && mounted) setState(() => _isConnected = true);

        try {
          final data = jsonDecode(message);
          
          // 1. HANDLE SPEAK COMMAND
          if (data['cmd'] == 'speak') {
             String text = data['text'];
             print("AI COMMAND: Speak -> $text");

             // --- FEATURE 9: URGENCY CODING & HAPTICS ---
             // If the backend tags the message as critical, buzz the phone first
             if (text.contains("[CRITICAL]")) {
                print("CRITICAL ALERT RECEIVED: Triggering Haptics");
                HapticFeedback.heavyImpact();
                HapticFeedback.heavyImpact();
                HapticFeedback.heavyImpact(); // tripple buzz for danger
    
                // Visual Alert
                if(mounted) setState(() => aiStatus = "DANGER");

                // Clean the text so the voice doesn't say the tag
                text = text.replaceFirst("[CRITICAL]", "Warning! ");          
             }
             // Update UI for normal speech
             else if(mounted) {
               setState(() {
                 aiStatus = "SPEAKING";
               });
             }
             _speak(text);
          }
          
          // 2. HANDLE INTERRUPT (Barge-In)
          else if (data['cmd'] == 'interrupt') {
            print("INTERRUPT COMMAND");
            flutterTts.stop(); // Stop speaking immediately
            if(mounted) setState(() => aiStatus = "INTERRUPTED");
          }
          
          // 3. HANDLE STATUS UPDATES
          else if (data['cmd'] == 'status') {
            if(mounted) {
              setState(() {
                aiStatus = data['state'].toString().toUpperCase();
              });
            }
          }

        } catch (e) {
          print("Error parsing server message: $e");
        }
      }, onError: (error) {
        print("WebSocket Error: $error");
        if(mounted) setState(() => _isConnected = false);
      }, onDone: () {
        print("WebSocket Closed");
        if(mounted) setState(() => _isConnected = false);
      });
    } catch (e) {
      print("Connection Failed: $e");
    }
  }
  
  Future<void> _speak(String text) async {
    await flutterTts.stop(); 
    if (text.isNotEmpty) {
      await flutterTts.speak(text);
    }
  }

  Future<void> _requestPermissions() async {
    await [Permission.camera, Permission.microphone].request();
    _initCameras();
  }

  // --- NEW: CAMERA SETUP LOGIC ---
  Future<void> _initCameras() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isNotEmpty) {
        _initializeCameraAtIndex(0);
      }
    } on CameraException catch (e) {
      print('Error finding cameras: $e');
    }
  }

  Future<void> _initializeCameraAtIndex(int index) async {
    if (_cameras.isEmpty) return;

    controller = CameraController(
      _cameras[index],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await controller!.initialize();
      if (!mounted) return;
      setState(() {
        _selectedCameraIndex = index;
        debugStatus = "Camera Ready. Tap Activate.";
      });
    } catch (e) {
      print("Camera init error: $e");
    }
  }

  void _switchCamera() async {
    if (_cameras.length < 2) return;

    // 1. STOP EVERYTHING IMMEDIATELY
    flutterTts.stop(); // Silence the voice
    _channel?.sink.close(); // Kill the connection to server (Flushes old buffers)
    setState(() {
      isStreaming = false;
      _isConnected = false; // Show offline icon briefly
    });

    int newIndex = (_selectedCameraIndex + 1) % _cameras.length;
    bool wasStreaming = isStreaming;
    
    // Stop stream before switching to prevent resource locks
    if (isStreaming) {
      await controller?.stopImageStream();
      setState(() => isStreaming = false);
    }
    
    await controller?.dispose();
    await _initializeCameraAtIndex(newIndex);

    await _connectWebSocket();

    // Resume stream if it was running
    if (wasStreaming) {
      _toggleStream();
    }
  }
  // -------------------------------

  void _toggleStream() {
    if (controller == null || !controller!.value.isInitialized) return;

    if (isStreaming) {
      controller!.stopImageStream();
      setState(() {
        isStreaming = false;
        debugStatus = "Stream Paused";
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

  // Starts listening to real-time location changes

  void _startLocationUpdates() async {
    print("Initializing Location Services...");
    
    // 1. Check if Location Service is enabled on the phone
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      print("Location services are disabled. Please turn on GPS.");
      return;
    }

    // 2. Check & Request Permissions
    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        print("Location permissions are denied.");
        return;
      }
    }
    
    if (permission == LocationPermission.deniedForever) {
      print("Location permissions are permanently denied.");
      return;
    }

    print(" Location Permission Granted. Starting Stream...");

    // 3. Start Streaming (High Accuracy)
    final LocationSettings locationSettings = LocationSettings(
      accuracy: LocationAccuracy.high,
      distanceFilter: 10, // Update every 10 meters
    );

    _positionStreamSubscription = Geolocator.getPositionStream(locationSettings: locationSettings)
        .listen((Position position) {
      print("New Location: ${position.latitude}, ${position.longitude}");
      
      if (mounted) {
        setState(() {
          _currentPosition = position;
        });
      }
    }, onError: (e) {
      print("Location Stream Error: $e");
    });
  }

  Future<void> _processFrame(CameraImage image) async {
    if (isProcessingFrame) return; 
    final now = DateTime.now();
    // Throttle: 1 frame every 1.5 seconds
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
            // --- FEATURE 5: CONTEXT INJECTION ---
            // We attach location data to the visual payload
            Map<String, double>? locationData;
            if(_currentPosition !=null) {
                locationData = {
                    'lat': _currentPosition!.latitude,
                    'lng': _currentPosition!.longitude
                };
            }
            
            _channel!.sink.add(jsonEncode({
                "image": base64Result,
                "timestamp": DateTime.now().millisecondsSinceEpoch,
                "location": locationData // Sending real GPS data or null
            }));
        }
        
        if(mounted) {
          setState(() {
            debugStatus = "Analyzing... ($aiStatus)"; 
          });
        }
      }
    } catch (e) {
      print("Error processing frame: $e");
    } finally {
      isProcessingFrame = false;
    }
  }

  @override
  void dispose() {
    _positionStreamSubscription?.cancel(); //Don't forget to cancel
    _channel?.sink.close();
    flutterTts.stop();
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Aura Vision"),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await FirebaseAuth.instance.signOut();
            },
          )
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: controller == null || !controller!.value.isInitialized
                ? const Center(child: CircularProgressIndicator())
                : Stack(
                    children: [
                      CameraPreview(controller!),
                      Positioned(
                        top: 20, right: 20,
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.black54, borderRadius: BorderRadius.circular(10)
                          ),
                          child: Text(aiStatus, style: const TextStyle(color: Colors.white)),
                        ),
                      )
                    ],
                  ),
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            color: Colors.black87,
            child: Column(
              children: [
                Text(
                  debugStatus,
                  style: const TextStyle(color: Colors.white, fontSize: 14, fontFamily: 'monospace'),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    // SWITCH CAMERA BUTTON
                    IconButton(
                      icon: const Icon(Icons.switch_camera, color: Colors.white, size: 30),
                      onPressed: _switchCamera,
                    ),
                    
                    ElevatedButton(
                      onPressed: _toggleStream,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: isStreaming? Colors.red : Colors.green,
                        minimumSize: const Size(150, 50),
                      ),
                      child: Text(
                        isStreaming ? "STOP" : "ACTIVATE",
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                    ),
                    
                    // STATUS ICON
                    Icon(
                      _isConnected ? Icons.cloud_done : Icons.cloud_off,
                      color: _isConnected ? Colors.blue : Colors.grey,
                      size: 30,
                    )
                  ],
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}

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
    final jpegBytes = img.encodeJpg(resized, quality: 70); 
    return base64Encode(jpegBytes);
  } catch (e) {
    return null;
  }
}