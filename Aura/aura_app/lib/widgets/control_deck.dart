import 'dart:ui';
import 'package:flutter/material.dart';

class ControlDeck extends StatelessWidget {
  final String statusText;
  final String aiStatus;
  final bool isStreaming;
  final bool isListening;
  final Animation<double> pulseAnimation;
  final VoidCallback onSwitchCamera;
  final VoidCallback onToggleStream;
  final VoidCallback onToggleMic;

  const ControlDeck({
    super.key,
    required this.statusText,
    required this.aiStatus,
    required this.isStreaming,
    required this.isListening,
    required this.pulseAnimation,
    required this.onSwitchCamera,
    required this.onToggleStream,
    required this.onToggleMic,
  });

  @override
  Widget build(BuildContext context) {
    Color statusColor = _getStatusColor(aiStatus);
    if (isListening) statusColor = Colors.purpleAccent;

    return Positioned(
      bottom: 30,
      left: 20,
      right: 20,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 1. AI STATUS ORB
          Container(
            height: 80,
            width: 80,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: statusColor.withOpacity(0.1),
              border: Border.all(color: statusColor.withOpacity(0.5), width: 2),
              boxShadow: [
                BoxShadow(
                  color: statusColor.withOpacity(0.3 * pulseAnimation.value),
                  blurRadius: 20 * pulseAnimation.value,
                  spreadRadius: 5 * pulseAnimation.value,
                )
              ],
            ),
            child: Icon(
              isListening
                  ? Icons.mic
                  : aiStatus == "DANGER"
                      ? Icons.warning_amber_rounded
                      : aiStatus == "SPEAKING"
                          ? Icons.graphic_eq
                          : aiStatus == "WATCHING"
                              ? Icons.remove_red_eye
                              : Icons.circle,
              color: statusColor,
              size: 32,
            ),
          ),

          const SizedBox(height: 20),

          // 2. SUBTITLES
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            child: Container(
              key: ValueKey<String>(statusText),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                statusText,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white, fontSize: 14),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ),

          const SizedBox(height: 20),

          // 3. HARDWARE CONTROLS (Cleaned up)
          ClipRRect(
            borderRadius: BorderRadius.circular(30),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 30),
                color: Colors.white.withOpacity(0.1),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // Switch Camera
                    IconButton(
                      icon: const Icon(Icons.switch_camera_rounded, color: Colors.white, size: 28),
                      onPressed: onSwitchCamera,
                    ),

                    // Main Action (Activate/Stop)
                    SizedBox(
                      height: 60, width: 60,
                      child: FloatingActionButton(
                        backgroundColor: isStreaming ? Colors.redAccent : Colors.greenAccent,
                        onPressed: onToggleStream,
                        elevation: 0,
                        child: Icon(
                          isStreaming ? Icons.stop_rounded : Icons.play_arrow_rounded,
                          color: Colors.black,
                          size: 32,
                        ),
                      ),
                    ),

                    // Microphone
                    GestureDetector(
                      onTap: onToggleMic,
                      child: CircleAvatar(
                        backgroundColor: isListening ? Colors.purpleAccent : Colors.transparent,
                        radius: 24,
                        child: Icon(
                          isListening ? Icons.mic : Icons.mic_none_rounded,
                          color: Colors.white,
                          size: 28,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
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