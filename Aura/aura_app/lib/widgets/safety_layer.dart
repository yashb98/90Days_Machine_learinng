import 'package:flutter/material.dart';

class SafetyLayer extends StatelessWidget {
  final String aiStatus;

  const SafetyLayer({super.key, required this.aiStatus});

  @override
  Widget build(BuildContext context) {
    // We use IgnorePointer so taps pass through to the camera/buttons below
    return IgnorePointer(
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        decoration: BoxDecoration(
          border: Border.all(
            // Only show red border if status is DANGER
            color: aiStatus == "DANGER" ? Colors.red.withOpacity(0.6) : Colors.transparent,
            width: 12,
          ),
        ),
      ),
    );
  }
}